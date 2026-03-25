import os
from typing import Callable, Optional

from PySide6 import QtCore

import Code
from Code.Z import Util
if __debug__:
    from Code import Debug
from Code.Engines import EngineResponse, EngineRun, Engines, Priorities
from Code.SQL import UtilSQL


class EngineManager:
    def __init__(self, engine, run_engine_params: EngineRun.RunEngineParams, with_cache: bool):

        self.engine_run: Optional[EngineRun.EngineRun] = None
        self.engine: Engines.Engine = engine

        self.starting_the_engine = False
        self.multipv: int = engine.multiPV
        self.priority: int = Priorities.priorities.normal

        self.run_engine_params = run_engine_params

        self._is_canceled = False
        self.elapsed_time = QtCore.QElapsedTimer()

        self.path_log: Optional[str] = None
        self.fichero_log: Optional[str] = None
        self.fichero_diag: Optional[str] = None  # 仅调试用
        self.diagnostic_context: Optional[str] = None  # 仅调试用
        self.diagnostic_session: Optional[str] = None  # 仅调试用

        self.mrm: Optional[EngineResponse.MultiEngineResponse] = None
        self.bestmove: Optional[str] = None
        self.depthchanged_connected_to: Optional[Callable] = None
        self.bestmove_connected_to: Optional[Callable] = None

        self.ms_refresh: int = 325

        self.allways_faster_mode = False

        self.cache_analysis = UtilSQL.DictBig() if with_cache else None

        self.enabled_emit_depth_changed = False
        self.enabled_emit_bestmove_found = False

        self.is_closed = False
        self.is_disabled = False  # util al analizar un movimiento que se espera hasta que termine

        self.active_loop: Optional[QtCore.QEventLoop] = None

        self.huella = Util.huella()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def disable(self):
        self.is_disabled = True

    def change_multipv(self, multipv: int):
        self.run_engine_params.multipv = multipv
        if self.engine_run:
            self.engine_run.set_multipv(multipv)

    def maximize_multipv(self):
        self.change_multipv(self.engine.maxMultiPV)

    def set_multipv_var(self, multipv_var: str | int):
        previo = self.engine.multiPV
        self.engine.set_multipv_var(multipv_var)
        if previo != self.engine.multiPV:
            self.change_multipv(self.engine.multiPV)

    def set_faster_mode(self):
        self.allways_faster_mode = True

    def set_option(self, name, value):
        if self.check_engine():
            self.engine_run.set_option(name, value)

    def set_priority_very_low(self):
        self.priority = Priorities.priorities.verylow

    def set_priority(self, priority: int):
        self.priority = priority

    @QtCore.Slot()
    def depth_changed(self):
        if (
            self.enabled_emit_depth_changed
            and self.depthchanged_connected_to is not None
            and self.engine_run is not None
            and self.engine_run.mrm is not None
        ):
            if mrm := self.engine_run.get_mrm():
                if self.depthchanged_connected_to(mrm) is False:
                    self.engine_run.stop()

    def connect_depthchanged(self, rutina):
        self.depthchanged_connected_to = rutina
        self.enabled_emit_depth_changed = rutina is not None

    @QtCore.Slot()
    def bestmove_found(self, bestmove):
        self.bestmove = bestmove
        if (
            self.enabled_emit_bestmove_found
            and self.bestmove_connected_to is not None
            and self.engine_run
            and self.engine_run.mrm
        ):
            self.bestmove_connected_to(bestmove)

    def connect_bestmove(self, rutina):
        self.bestmove_connected_to = rutina
        self.enabled_emit_bestmove_found = rutina is not None

    def disconnect(self):
        self.bestmove_connected_to = None
        self.depthchanged_connected_to = None
        self.enabled_emit_depth_changed = False
        self.enabled_emit_bestmove_found = False

    def check_engine(self) -> bool:
        if self.starting_the_engine:
            return True
        if self.engine_run is not None:
            return True

        self.starting_the_engine = True

        if __debug__:
            Debug.prln(f"EngineManager.open called for {self.engine.name} {self.huella}", color="yellow")

        config_enginerun = EngineRun.StartEngineParams()
        config_enginerun.name = self.engine.name
        config_enginerun.path_exe = self.engine.ejecutable()
        config_enginerun.args = self.engine.argumentos()
        config_enginerun.li_options_uci = self.engine.liUCI
        config_enginerun.faster_mode_always = self.allways_faster_mode
        config_enginerun.priority = self.priority
        if self.engine.emulate_movetime:
            config_enginerun.emulate_movetime = True
        if Code.list_engine_managers.with_logs:
            config_enginerun.path_log = self.set_path_log()
        if self.fichero_log:  # 仅调试用
            config_enginerun.path_log = self.fichero_log  # 仅调试用
        if self.fichero_diag:  # 仅调试用
            config_enginerun.path_diag = self.fichero_diag  # 仅调试用

        self.engine_run = EngineRun.EngineRun(config_enginerun)
        if self.engine_run.state == EngineRun.EngineState.INVALID_ENGINE:
            self.starting_the_engine = False
            self.close()
            return False
        self.engine_run.bestmove_found.connect(self.bestmove_found)
        self.engine_run.depth_changed.connect(self.depth_changed)

        Code.list_engine_managers.append(self)

        if self.multipv > 0:
            self.engine_run.set_multipv(self.multipv)

        self.starting_the_engine = False
        return True

    def stop(self):
        if self.engine_run:
            if __debug__:
                Debug.prln(f"EngineManager.stop() called for {self.engine.name}", color="yellow")
            self.engine_run.stop()

    def enable_diagnostics(self, context: str, session: Optional[str] = None):  # 仅调试用
        folder = Util.opj(Code.configuration.paths.folder_userdata(), "EngineLogs", "PlayAgainstEngine")  # 仅调试用
        if not os.path.isdir(folder):  # 仅调试用
            Util.create_folder(folder)  # 仅调试用

        safe_context = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in context)  # 仅调试用
        safe_engine = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in self.engine.name)  # 仅调试用
        safe_session = session or self.huella  # 仅调试用
        base = Util.opj(folder, f"{safe_context}_{safe_engine}_{safe_session}")  # 仅调试用

        self.diagnostic_context = context  # 仅调试用
        self.diagnostic_session = safe_session  # 仅调试用
        self.fichero_log = f"{base}.uci.log"  # 仅调试用
        self.fichero_diag = f"{base}.diag.log"  # 仅调试用

        if self.engine_run:  # 仅调试用
            self.engine_run.log_open(self.fichero_log)  # 仅调试用
            self.engine_run.diag_open(self.fichero_diag)  # 仅调试用
            self.engine_run.log_diagnostic(  # 仅调试用
                "diagnostic-session-attached",  # 仅调试用
                context=self.diagnostic_context,  # 仅调试用
                session=self.diagnostic_session,  # 仅调试用
            )  # 仅调试用

    def log_diagnostic(self, event: str, **info):  # 仅调试用
        if self.engine_run:  # 仅调试用
            self.engine_run.log_diagnostic(event, **info)  # 仅调试用

    def engine_snapshot(self) -> dict:  # 仅调试用
        if self.engine_run:  # 仅调试用
            return self.engine_run.diagnostic_snapshot()  # 仅调试用
        return {  # 仅调试用
            "engine": self.engine.name,  # 仅调试用
            "state": "not-started",  # 仅调试用
            "session": self.diagnostic_session,  # 仅调试用
        }  # 仅调试用

    def close(self):
        if self.is_closed:
            return
        if __debug__:
            Debug.prln(f"EngineManager.close() called for {self.engine.name} {self.huella}", color="yellow")
        self.is_closed = True

        if self.active_loop:
            try:
                self.active_loop.quit()
            except:
                pass
            self.active_loop = None

        if self.engine_run:
            try:
                self.engine_run.close()
            except:
                pass
            self.engine_run = None

        if self.cache_analysis is not None:
            try:
                self.cache_analysis.close()
            except:
                pass
            self.cache_analysis = None

        if Code.list_engine_managers:
            Code.list_engine_managers.cleanup_closed()
            
    def set_path_log(self):
        carpeta = Util.opj(Code.configuration.paths.folder_userdata(), "EngineLogs")
        if not os.path.isdir(carpeta):
            Util.create_folder(carpeta)
        plantlog = f"{Util.opj(carpeta, self.engine.name)}_%05d"
        pos = 1
        path_log = plantlog % pos

        while os.path.isfile(path_log):
            pos += 1
            path_log = plantlog % pos

        self.path_log = path_log
            
        return path_log

    def log_open(self):
        if self.path_log:
            return
        self.set_path_log()

        if self.engine_run:
            self.engine_run.log_open(self.path_log)

    def log_close(self):
        self.path_log = None
        if self.engine_run:
            self.engine_run.log_close()

    def li_uci(self) -> list:
        if self.check_engine():
            return self.engine_run.li_uci
        return []

    def get_current_mrm(self) -> EngineResponse.MultiEngineResponse | None:
        if self.engine_run:
            return self.engine_run.get_mrm()
        return None

    def is_run_fixed(self):
        return self.run_engine_params.is_fixed()

    def is_run_fast(self):
        return self.run_engine_params.is_fast()

    def mstime_run(self):
        return self.run_engine_params.fixed_ms

    def update_time_run(self, time_secs_white, time_secs_black, inc_time_secs_move):
        self.run_engine_params.update_var_time(time_secs_white, time_secs_black, inc_time_secs_move)

    @property
    def name(self):
        return self.engine.name
