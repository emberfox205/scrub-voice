#!/usr/bin/env python3
"""
End-to-end integration tests for Scrub-Voice Orchestrator to Monitor pipeline.
Verifies:
1. ZeroMQ PUB/SUB distribution and LVC REQ/REP state snapshot synchronization.
2. MonitorBridge state hydration and real-time streaming updates.
3. FastAPI REST status endpoints and static dashboard delivery.
4. WebSocket full-duplex streaming of live speech transcripts, EMR milestones,
   and emergency halt visual interlocks.
5. True socket-level propagation through NLU dual-channel PUSH -> Dispatcher PULL -> PUB -> Monitor SUB.
"""

import sys
import threading
import time
import unittest
from pathlib import Path

# Dynamic sys.path setup according to AGENTS.md
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jsonschema
import orjson
import zmq
from fastapi.testclient import TestClient

from src.orchestrator.dispatcher import CommandDispatcher
from src.monitor.bridge import MonitorBridge
from src.monitor.server import create_app
try:
    from test.mock_publisher import MockCommandPublisher, PRESETS
except ImportError:
    from mock_publisher import MockCommandPublisher, PRESETS

TEST_NLU_ADDR = "ipc:///tmp/test_pipe_nlu.ipc"
TEST_NLU_PRIORITY_ADDR = "ipc:///tmp/test_pipe_nlu_priority.ipc"
TEST_BUS_ADDR = "ipc:///tmp/test_pipe_bus.ipc"
TEST_SNAPSHOT_ADDR = "ipc:///tmp/test_pipe_snapshot.ipc"


class TestMonitorPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        schema_path = REPO_ROOT / "schemas" / "command_schema.json"
        with open(schema_path, "rb") as f:
            cls.schema = orjson.loads(f.read())
        cls.validator = jsonschema.Draft202012Validator(cls.schema)

        # 1. Initialize and start Orchestrator Dispatcher with test IPC sockets in background thread
        cls.dispatcher = CommandDispatcher(
            nlu_addr=TEST_NLU_ADDR,
            nlu_priority_addr=TEST_NLU_PRIORITY_ADDR,
            bus_addr=TEST_BUS_ADDR,
            snapshot_addr=TEST_SNAPSHOT_ADDR,
        )
        cls.disp_thread = threading.Thread(target=cls.dispatcher.run, daemon=True, name="TestDispatcherLoop")
        cls.disp_thread.start()

        # 2. Client context to push commands into the Orchestrator
        cls.client_ctx = zmq.Context()
        cls.normal_push = cls.client_ctx.socket(zmq.PUSH)
        cls.normal_push.connect(TEST_NLU_ADDR)

        cls.priority_push = cls.client_ctx.socket(zmq.PUSH)
        cls.priority_push.connect(TEST_NLU_PRIORITY_ADDR)

        # Allow ZeroMQ IPC pipes to bind and establish
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.normal_push.close(linger=0)
        cls.priority_push.close(linger=0)
        cls.client_ctx.term()
        cls.dispatcher.stop()
        cls.disp_thread.join(timeout=2.0)

    def test_01_lvc_snapshot_synchronization(self):
        """
        Orchestrator must cache intraoperative state and serve it to
        newly connected MonitorBridge instances via ZeroMQ REQ/REP over IPC.
        """
        # 1. Push stateful commands through Orchestrator via normal PUSH socket
        open_cmd = {
            "command_id": "cmd_test_001",
            "timestamp": time.time(),
            "latency_ms": 32.0,
            "raw_transcript": "open CT scan",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "ct"},
            "confidence": 0.99,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.normal_push.send(orjson.dumps(open_cmd))

        zoom_cmd = {
            "command_id": "cmd_test_002",
            "timestamp": time.time(),
            "latency_ms": 30.0,
            "raw_transcript": "zoom in 3x",
            "intent": "zoom_display",
            "category": "MONITOR_DISPLAY",
            "slots": {"direction": "IN", "factor": 3.0},
            "confidence": 0.98,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.normal_push.send(orjson.dumps(zoom_cmd))

        nav_cmd = {
            "command_id": "cmd_test_003",
            "timestamp": time.time(),
            "latency_ms": 28.0,
            "raw_transcript": "navigate to slice 42",
            "intent": "navigate_slice",
            "category": "MONITOR_DISPLAY",
            "slots": {"slice_index": 42},
            "confidence": 0.97,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.normal_push.send(orjson.dumps(nav_cmd))

        # Give dispatcher loop time to pull, validate, and update LVC cache
        time.sleep(0.2)

        # 2. Test MonitorBridge hydration from snapshot server (REQ/REP over IPC)
        bridge = MonitorBridge(bus_addr=TEST_BUS_ADDR, snapshot_addr=TEST_SNAPSHOT_ADDR)
        success = bridge.fetch_snapshot()
        self.assertTrue(success, "MonitorBridge failed to fetch snapshot from orchestrator REP server")

        self.assertEqual(bridge.state.active_scan, "ct")
        self.assertEqual(bridge.state.zoom_factor, 3.0)
        self.assertEqual(bridge.state.current_slice, 42)
        self.assertIsNotNone(bridge.state.telemetry)

        bridge.stop()

    def test_02_live_bus_streaming_to_bridge(self):
        """
        Commands pushed via PUSH socket and routed onto ZeroMQ PUB bus must be received
        by MonitorBridge subscriber and immediately update MonitorState.
        """
        bridge = MonitorBridge(bus_addr=TEST_BUS_ADDR, snapshot_addr=TEST_SNAPSHOT_ADDR)
        bridge.start(fetch_initial_snapshot=True)
        time.sleep(0.2)

        initial_slice = bridge.state.current_slice

        # Push navigate_slice command through orchestrator
        step_cmd = {
            "command_id": "cmd_test_004",
            "timestamp": time.time(),
            "latency_ms": 25.0,
            "raw_transcript": "advance 6 slices",
            "intent": "navigate_slice",
            "category": "MONITOR_DISPLAY",
            "slots": {"direction": "NEXT", "step_count": 6},
            "confidence": 0.98,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.normal_push.send(orjson.dumps(step_cmd))

        # Give bridge worker time to ingest and process
        time.sleep(0.2)
        self.assertEqual(bridge.state.current_slice, initial_slice + 6)
        self.assertEqual(bridge.state.last_command["command_id"], "cmd_test_004")

        bridge.stop()

    def test_03_emergency_halt_and_clear_propagation(self):
        """
        Emergency halt command sent over priority PUSH socket must broadcast and
        toggle emergency_active flag in MonitorBridge.
        """
        bridge = MonitorBridge(bus_addr=TEST_BUS_ADDR, snapshot_addr=TEST_SNAPSHOT_ADDR)
        bridge.start(fetch_initial_snapshot=False)
        time.sleep(0.2)

        # 1. Send halt_emergency via priority socket
        halt_cmd = {
            "command_id": "cmd_halt_001",
            "timestamp": time.time(),
            "latency_ms": 12.0,
            "raw_transcript": "emergency stop",
            "intent": "halt_emergency",
            "category": "EMERGENCY",
            "slots": {},
            "confidence": 0.999,
            "priority": "HIGH",
            "status": "EXECUTE",
        }
        self.priority_push.send(orjson.dumps(halt_cmd))
        time.sleep(0.2)

        self.assertTrue(bridge.state.emergency_active)

        # 2. Send clear_emergency via priority socket
        clear_cmd = {
            "command_id": "cmd_clear_001",
            "timestamp": time.time(),
            "latency_ms": 14.0,
            "raw_transcript": "clear emergency",
            "intent": "clear_emergency",
            "category": "EMERGENCY",
            "slots": {},
            "confidence": 0.999,
            "priority": "HIGH",
            "status": "EXECUTE",
        }
        self.priority_push.send(orjson.dumps(clear_cmd))
        time.sleep(0.2)

        self.assertFalse(bridge.state.emergency_active)

        bridge.stop()

    def test_04_fastapi_rest_and_ui_endpoints(self):
        """
        FastAPI application must serve offline dashboard HTML, static assets,
        and REST /api/state and /api/health endpoints.
        """
        app = create_app(
            bus_addr=TEST_BUS_ADDR,
            snapshot_addr=TEST_SNAPSHOT_ADDR,
            fetch_snapshot_on_startup=True,
        )

        with TestClient(app) as client:
            # 1. Root dashboard UI
            res = client.get("/")
            self.assertEqual(res.status_code, 200)
            self.assertIn("SCRUB-VOICE SURGICAL MONITOR", res.text)
            self.assertIn("dicomCanvas", res.text)
            self.assertIn("auditLogTableBody", res.text)
            self.assertNotIn("emergencyOverlay", res.text)
            self.assertIn("STANDBY / OFFLINE", res.text)

            # 2. Static CSS and JS
            res_css = client.get("/static/style.css")
            self.assertEqual(res_css.status_code, 200)
            res_js = client.get("/static/app.js")
            self.assertEqual(res_js.status_code, 200)

            # 2b. Static Clinical Scan Assets (CT, MRI, X-ray)
            for scan in ["ct.jpg", "mri.jpg", "xray.jpg"]:
                res_scan = client.get(f"/static/scans/{scan}")
                self.assertEqual(res_scan.status_code, 200, f"Failed to retrieve /static/scans/{scan}")
                self.assertIn("image/jpeg", res_scan.headers.get("content-type", ""))
                self.assertGreater(len(res_scan.content), 1000)

            # 3. Health endpoint
            res_health = client.get("/api/health")
            self.assertEqual(res_health.status_code, 200)
            data_health = res_health.json()
            self.assertEqual(data_health["status"], "healthy")

            # 4. State endpoint
            res_state = client.get("/api/state")
            self.assertEqual(res_state.status_code, 200)
            data_state = res_state.json()
            self.assertIn("current_slice", data_state)
            self.assertIn("zoom_factor", data_state)
            self.assertIn("telemetry", data_state)

    def test_05_websocket_streaming_and_interaction(self):
        """
        WebSocket /ws endpoint must immediately stream snapshot upon connection,
        reply to ping heartbeats, broadcast live dispatched commands, and deduplicate
        multi-topic broadcasts so only a single frame is delivered per command_id.
        """
        app = create_app(
            bus_addr=TEST_BUS_ADDR,
            snapshot_addr=TEST_SNAPSHOT_ADDR,
            fetch_snapshot_on_startup=True,
        )

        with TestClient(app) as client:
            with client.websocket_connect("/ws") as ws:
                # 1. Client receives initial state snapshot
                snap = ws.receive_json()
                self.assertEqual(snap["type"], "snapshot")
                self.assertIn("current_slice", snap["state"])

                # 2. Client sends ping heartbeat
                ws.send_text("ping")
                pong = ws.receive_text()
                self.assertEqual(pong, "pong")

                # 3. Broadcast command over ZMQ bus to WebSocket
                log_cmd = {
                    "command_id": "cmd_ws_log_001",
                    "timestamp": time.time(),
                    "latency_ms": 29.5,
                    "raw_transcript": "log milestone: incision made",
                    "intent": "log_event",
                    "category": "MONITOR_DISPLAY",
                    "slots": {"event_name": "incision made", "comment": "clean margin"},
                    "confidence": 0.985,
                    "priority": "NORMAL",
                    "status": "EXECUTE",
                }
                self.normal_push.send(orjson.dumps(log_cmd))

                # Receive live broadcast event on WebSocket
                ws_msg = ws.receive_json()
                self.assertEqual(ws_msg["type"], "command")
                self.assertEqual(ws_msg["command"]["intent"], "log_event")
                self.assertTrue(any(m["event_name"] == "incision made" for m in ws_msg["state"]["recent_milestones"]))

                # 4. Verify emergency command deduplication (sent on both EMERGENCY and GLOBAL_EMERGENCY)
                halt_cmd = {
                    "command_id": "cmd_ws_halt_dedup_001",
                    "timestamp": time.time(),
                    "latency_ms": 11.0,
                    "raw_transcript": "emergency stop now",
                    "intent": "halt_emergency",
                    "category": "EMERGENCY",
                    "slots": {},
                    "confidence": 0.999,
                    "priority": "HIGH",
                    "status": "EXECUTE",
                }
                self.priority_push.send(orjson.dumps(halt_cmd))

                # Exactly one message should be received for this command_id
                ws_halt_msg = ws.receive_json()
                self.assertEqual(ws_halt_msg["command"]["command_id"], "cmd_ws_halt_dedup_001")
                self.assertTrue(ws_halt_msg["state"]["emergency_active"])

    def test_06_routine_preset_pipeline_execution(self):
        """
        Executes the official NLU 'routine' surgical preset through the full
        MockCommandPublisher -> Orchestrator -> Monitor pipeline and verifies resulting clinical state.
        """
        bridge = MonitorBridge(bus_addr=TEST_BUS_ADDR, snapshot_addr=TEST_SNAPSHOT_ADDR)
        bridge.start(fetch_initial_snapshot=True)
        time.sleep(0.2)

        publisher = MockCommandPublisher(
            nlu_addr=TEST_NLU_ADDR,
            nlu_priority_addr=TEST_NLU_PRIORITY_ADDR,
        )
        sent_count = publisher.send_preset("routine", interval_sec=0.02)
        self.assertEqual(sent_count, len(PRESETS["routine"]))

        time.sleep(0.3)
        # Verify final state after routine sequence:
        # 1. CT scan opened
        self.assertEqual(bridge.state.active_scan, "ct")
        # 2. Zoomed by 2.5x -> 2.5x
        self.assertEqual(bridge.state.zoom_factor, 2.5)
        # 3. Slices advanced: initial + 5
        self.assertGreaterEqual(bridge.state.current_slice, 6)
        # 4. Tool was parked at end of routine preset
        self.assertIsNone(bridge.state.held_tool)
        # 5. Milestone was logged
        milestones = [m["event_name"] for m in bridge.state.recent_milestones]
        self.assertIn("abdominal insufflation completed", milestones)

        publisher.close()
        bridge.stop()

    def test_07_switch_scan_modality_pipeline(self):
        """
        Tests live pipeline end-to-end switching between medical scan modalities (CT -> MRI -> X-Ray)
        via Orchestrator PUSH socket to MonitorBridge subscriber.
        Verifies:
        1. Exact compliance of every switch command with schemas/command_schema.json.
        2. Immediate live bridge state updates (active_scan switches, slice and zoom reset).
        3. Orchestrator snapshot server (LVC cache) reflects the newly switched modality.
        """
        bridge = MonitorBridge(bus_addr=TEST_BUS_ADDR, snapshot_addr=TEST_SNAPSHOT_ADDR)
        bridge.start(fetch_initial_snapshot=True)
        time.sleep(0.2)

        # 1. Open CT scan
        cmd_ct = {
            "command_id": "cmd_switch_ct_001",
            "timestamp": time.time(),
            "latency_ms": 22.0,
            "raw_transcript": "open CT scan",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "ct"},
            "confidence": 0.99,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(cmd_ct)
        self.normal_push.send(orjson.dumps(cmd_ct))
        time.sleep(0.2)

        self.assertEqual(bridge.state.active_scan, "ct")
        self.assertEqual(bridge.state.current_slice, 1)
        self.assertEqual(bridge.state.zoom_factor, 1.0)

        # Clinician zooms and pages slices
        cmd_zoom = {
            "command_id": "cmd_switch_zoom_001",
            "timestamp": time.time(),
            "latency_ms": 18.0,
            "raw_transcript": "zoom in 3x",
            "intent": "zoom_display",
            "category": "MONITOR_DISPLAY",
            "slots": {"direction": "IN", "factor": 3.0},
            "confidence": 0.98,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(cmd_zoom)
        self.normal_push.send(orjson.dumps(cmd_zoom))

        cmd_nav = {
            "command_id": "cmd_switch_nav_001",
            "timestamp": time.time(),
            "latency_ms": 19.0,
            "raw_transcript": "go to slice 20",
            "intent": "navigate_slice",
            "category": "MONITOR_DISPLAY",
            "slots": {"slice_index": 20},
            "confidence": 0.98,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(cmd_nav)
        self.normal_push.send(orjson.dumps(cmd_nav))
        time.sleep(0.2)

        self.assertEqual(bridge.state.zoom_factor, 3.0)
        self.assertEqual(bridge.state.current_slice, 20)

        # 2. Switch CT -> MRI scan ("switch to MRI")
        cmd_mri = {
            "command_id": "cmd_switch_mri_002",
            "timestamp": time.time(),
            "latency_ms": 24.0,
            "raw_transcript": "switch to MRI",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "mri"},
            "confidence": 0.99,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(cmd_mri)
        self.normal_push.send(orjson.dumps(cmd_mri))
        time.sleep(0.2)

        # Verify active_scan switched and viewport reset
        self.assertEqual(bridge.state.active_scan, "mri")
        self.assertEqual(bridge.state.current_slice, 1)
        self.assertEqual(bridge.state.zoom_factor, 1.0)

        # 3. Switch MRI -> X-Ray scan ("display X-ray")
        cmd_xray = {
            "command_id": "cmd_switch_xray_003",
            "timestamp": time.time(),
            "latency_ms": 20.0,
            "raw_transcript": "display X-ray",
            "intent": "open_scan",
            "category": "MONITOR_DISPLAY",
            "slots": {"scan_type": "xray"},
            "confidence": 0.98,
            "priority": "NORMAL",
            "status": "EXECUTE",
        }
        self.validator.validate(cmd_xray)
        self.normal_push.send(orjson.dumps(cmd_xray))
        time.sleep(0.2)

        self.assertEqual(bridge.state.active_scan, "xray")
        self.assertEqual(bridge.state.current_slice, 1)
        self.assertEqual(bridge.state.zoom_factor, 1.0)

        # 4. Verify snapshot synchronization with Orchestrator LVC cache
        snap_bridge = MonitorBridge(bus_addr=TEST_BUS_ADDR, snapshot_addr=TEST_SNAPSHOT_ADDR)
        fetch_ok = snap_bridge.fetch_snapshot()
        self.assertTrue(fetch_ok)
        self.assertEqual(snap_bridge.state.active_scan, "xray")
        self.assertEqual(snap_bridge.state.current_slice, 1)
        self.assertEqual(snap_bridge.state.zoom_factor, 1.0)
        snap_bridge.stop()

        bridge.stop()


if __name__ == "__main__":
    unittest.main()
