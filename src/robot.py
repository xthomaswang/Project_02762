"""
OT-2 Robot Control via HTTP API.

Ported from OT2_functions.py — globals converted to OT2Robot class attributes.
"""

import requests
from typing import Dict, Any, Optional, Tuple


class OT2Robot:
    """Interface to a single OT-2 robot over its HTTP API."""

    HEADERS = {"Opentrons-Version": "*"}

    def __init__(self, ip: str = "169.254.8.56", port: int = 31950):
        self.ip = ip
        self.port = port
        self.base_url = f"http://{ip}:{port}"
        self.commands_url: Optional[str] = None
        self.pipette_id: Optional[str] = None
        self.labware_by_slot: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Run management
    # ------------------------------------------------------------------

    def create_run(self) -> Tuple[str, str]:
        """Create a new run and return (run_id, commands_url)."""
        runs_url = f"{self.base_url}/runs"
        print(f"[OT] POST {runs_url}")
        r = requests.post(url=runs_url, headers=self.HEADERS)
        r.raise_for_status()

        data = r.json()["data"]
        run_id = data["id"]
        self.commands_url = f"{runs_url}/{run_id}/commands"
        print(f"[OT] Run created: {run_id}")
        return run_id, self.commands_url

    def reconnect_last_run(self, prefer_mount: str = "right") -> Dict[str, Any]:
        """Reconnect to the most recent run on the robot."""
        runs_url = f"{self.base_url}/runs"
        print(f"[OT] GET {runs_url}")
        r = requests.get(runs_url, headers=self.HEADERS)
        r.raise_for_status()

        runs = r.json().get("data", [])
        if not runs:
            raise RuntimeError("No runs found on the robot.")

        last_run = runs[-1]
        run_id = last_run["id"]
        self.commands_url = f"{runs_url}/{run_id}/commands"
        print(f"[OT] Reconnected to run: {run_id}")

        # Recover pipette_id
        self.pipette_id = None
        pipettes = last_run.get("pipettes", [])
        chosen = None
        for p in pipettes:
            if p.get("mount") == prefer_mount:
                chosen = p
                break
        if chosen is None and pipettes:
            chosen = pipettes[0]
        if chosen is not None:
            self.pipette_id = chosen.get("id")
            print(f"[OT] Recovered pipette_id: {self.pipette_id}")

        # Recover labware mapping
        self.labware_by_slot.clear()
        for lw in last_run.get("labware", []):
            lw_id = lw.get("id")
            location = lw.get("location") or {}
            slot = location.get("slotName") or location.get("slot_name")
            if lw_id and slot:
                self.labware_by_slot[slot] = lw_id
        print(f"[OT] LABWARE_BY_SLOT restored: {self.labware_by_slot}")

        return {
            "run_id": run_id,
            "pipette_id": self.pipette_id,
            "labware_by_slot": dict(self.labware_by_slot),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_command(self, command_dict: dict, wait: bool = True) -> dict:
        if not self.commands_url:
            raise RuntimeError("No active run. Call create_run() first.")
        params = {"waitUntilComplete": True} if wait else None
        r = requests.post(
            url=self.commands_url,
            headers=self.HEADERS,
            json=command_dict,
            params=params,
        )
        if not r.ok:
            print(f"\n[OT][HTTP ERROR] {r.status_code}")
            try:
                print(f"[OT][RESPONSE TEXT]: {r.text}")
            except Exception:
                pass
            r.raise_for_status()
        return r.json()["data"]

    def _require_pipette(self):
        if self.pipette_id is None:
            raise RuntimeError("Pipette not loaded.")

    # ------------------------------------------------------------------
    # Labware helpers
    # ------------------------------------------------------------------

    def get_labware_id_by_slot(self, slot_name: str) -> str:
        if slot_name in self.labware_by_slot:
            return self.labware_by_slot[slot_name]
        raise KeyError(
            f"No labwareId for slot '{slot_name}'. "
            f"Known slots: {list(self.labware_by_slot.keys())}"
        )

    # ------------------------------------------------------------------
    # Equipment loading
    # ------------------------------------------------------------------

    def load_equipment(self, equipment_type: int, equipment_name: str,
                       slot_name: Optional[str] = None) -> str:
        """
        Load pipette or labware.
        equipment_type: 0 -> pipette, 1 -> labware
        """
        if equipment_type == 0:
            cmd = {
                "data": {
                    "commandType": "loadPipette",
                    "params": {"pipetteName": equipment_name, "mount": "right"},
                    "intent": "setup",
                }
            }
            print(f"[OT] loadPipette: {equipment_name} (right)")
            data = self._post_command(cmd, wait=True)
            self.pipette_id = data["result"]["pipetteId"]
            print(f"[OT] Pipette ID: {self.pipette_id}")
            return self.pipette_id

        elif equipment_type == 1:
            if slot_name is None:
                raise ValueError("slot_name is required for labware")
            cmd = {
                "data": {
                    "commandType": "loadLabware",
                    "params": {
                        "location": {"slotName": slot_name},
                        "loadName": equipment_name,
                        "namespace": "opentrons",
                        "version": 1,
                    },
                    "intent": "setup",
                }
            }
            print(f"[OT] loadLabware: {equipment_name} in slot {slot_name}")
            data = self._post_command(cmd, wait=True)
            labware_id = data["result"]["labwareId"]
            print(f"[OT] Labware ID: {labware_id}")
            self.labware_by_slot[slot_name] = labware_id
            return labware_id

        else:
            raise ValueError("equipment_type must be 0 (pipette) or 1 (labware)")

    # ------------------------------------------------------------------
    # Motion & liquid handling
    # ------------------------------------------------------------------

    def pick_up(self, tiprack_id: str, wellname: str = "A1",
                offset: Optional[Tuple[float, ...]] = None):
        self._require_pipette()
        off = offset or (0, 0, 0)
        cmd = {
            "data": {
                "commandType": "pickUpTip",
                "params": {
                    "labwareId": tiprack_id,
                    "wellName": wellname,
                    "wellLocation": {
                        "origin": "top",
                        "offset": {"x": off[0], "y": off[1], "z": off[2]},
                    },
                    "pipetteId": self.pipette_id,
                },
                "intent": "setup",
            }
        }
        print(f"[OT] pickUpTip: labware={tiprack_id}, well={wellname}, offset={off}")
        self._post_command(cmd, wait=True)

    def move(self, labware_id: str, wellname: str = "A1",
             offset: Optional[Tuple[float, ...]] = None):
        self._require_pipette()
        off = offset or (0, 0, 0)
        cmd = {
            "data": {
                "commandType": "moveToWell",
                "params": {
                    "labwareId": labware_id,
                    "wellName": wellname,
                    "wellLocation": {
                        "origin": "top",
                        "offset": {"x": off[0], "y": off[1], "z": off[2]},
                    },
                    "pipetteId": self.pipette_id,
                },
                "intent": "setup",
            }
        }
        print(f"[OT] moveToWell: labware={labware_id}, well={wellname}, offset={off}")
        self._post_command(cmd, wait=True)

    def drop_tips(self, tiprack_id: Optional[str] = None, wellname: str = "A1",
                  offset: Optional[Tuple[float, ...]] = None):
        self._require_pipette()
        off = offset or (0, 0, 0)
        labware_id = tiprack_id or "fixedTrash"
        cmd = {
            "data": {
                "commandType": "dropTip",
                "params": {
                    "labwareId": labware_id,
                    "wellName": wellname,
                    "wellLocation": {
                        "origin": "top",
                        "offset": {"x": off[0], "y": off[1], "z": off[2]},
                    },
                    "pipetteId": self.pipette_id,
                },
                "intent": "setup",
            }
        }
        print(f"[OT] dropTip: labware={labware_id}, well={wellname}, offset={off}")
        self._post_command(cmd, wait=True)

    def unload_to_trash(self, wellname: str = "A1",
                        offset: Optional[Tuple[float, ...]] = None):
        self._require_pipette()
        off = offset or (0, 0, 0)
        print(f"[OT] unload_to_trash: fixedTrash, well={wellname}, offset={off}")
        move_cmd = {
            "data": {
                "commandType": "moveToAddressableAreaForDropTip",
                "params": {
                    "pipetteId": self.pipette_id,
                    "addressableAreaName": "fixedTrash",
                    "wellName": wellname,
                    "wellLocation": {
                        "origin": "default",
                        "offset": {"x": off[0], "y": off[1], "z": off[2]},
                    },
                    "alternateDropLocation": False,
                },
                "intent": "setup",
            }
        }
        self._post_command(move_cmd, wait=True)
        drop_cmd = {
            "data": {
                "commandType": "dropTipInPlace",
                "params": {"pipetteId": self.pipette_id},
                "intent": "setup",
            }
        }
        self._post_command(drop_cmd, wait=True)

    def aspirate(self, volume: float, labware_id: str, wellname: str,
                 offset: Optional[Tuple[float, ...]] = None,
                 origin: str = "bottom", flow_rate: float = 150.0):
        self._require_pipette()
        if labware_id is None or wellname is None:
            raise ValueError("aspirate() requires labware_id and wellname.")
        off = offset or (0, 0, 1)
        cmd = {
            "data": {
                "commandType": "aspirate",
                "params": {
                    "pipetteId": self.pipette_id,
                    "volume": volume,
                    "flowRate": flow_rate,
                    "labwareId": labware_id,
                    "wellName": wellname,
                    "wellLocation": {
                        "origin": origin,
                        "offset": {"x": off[0], "y": off[1], "z": off[2]},
                    },
                },
                "intent": "setup",
            }
        }
        print(f"[OT] aspirate: {volume}uL from {wellname}, origin={origin}, offset={off}")
        self._post_command(cmd, wait=True)

    def dispense(self, volume: float, labware_id: str, wellname: str,
                 offset: Optional[Tuple[float, ...]] = None,
                 origin: str = "bottom", flow_rate: float = 150.0):
        self._require_pipette()
        if labware_id is None or wellname is None:
            raise ValueError("dispense() requires labware_id and wellname.")
        off = offset or (0, 0, 1)
        cmd = {
            "data": {
                "commandType": "dispense",
                "params": {
                    "pipetteId": self.pipette_id,
                    "volume": volume,
                    "flowRate": flow_rate,
                    "labwareId": labware_id,
                    "wellName": wellname,
                    "wellLocation": {
                        "origin": origin,
                        "offset": {"x": off[0], "y": off[1], "z": off[2]},
                    },
                },
                "intent": "setup",
            }
        }
        print(f"[OT] dispense: {volume}uL into {wellname}, origin={origin}, offset={off}")
        self._post_command(cmd, wait=True)

    def blow_out(self, labware_id: Optional[str] = None,
                 wellname: Optional[str] = None,
                 offset: Optional[Tuple[float, ...]] = None):
        self._require_pipette()
        params: Dict[str, Any] = {"pipetteId": self.pipette_id, "flowRate": 100.0}
        if labware_id is not None or wellname is not None:
            if not (labware_id and wellname):
                raise ValueError("blow_out() needs both labware_id and wellname, or neither.")
            off = offset or (0, 0, 0)
            params.update({
                "labwareId": labware_id,
                "wellName": wellname,
                "wellLocation": {
                    "origin": "top",
                    "offset": {"x": off[0], "y": off[1], "z": off[2]},
                },
            })
        cmd = {"data": {"commandType": "blowout", "params": params, "intent": "setup"}}
        print("[OT] blow_out")
        self._post_command(cmd, wait=True)

    def pause_run(self, message: str = "Tip check failed."):
        cmd = {
            "data": {
                "commandType": "pause",
                "params": {"message": message},
                "intent": "protocol",
            }
        }
        print(f"[OT] pause: {message}")
        self._post_command(cmd, wait=True)

    def home(self):
        cmd = {"data": {"commandType": "home", "params": {}, "intent": "setup"}}
        print("[OT] home: Returning to home position")
        self._post_command(cmd, wait=True)
