# -*- coding: utf-8 -*-
"""
Created on Sun Nov 16 18:37:56 2025

@author: Divya
"""
import threading
import time
from queue import Queue, Empty
from typing import Dict, Tuple, Iterable, Optional, Iterator, List
import numpy as np

# {station_id: (X_train, Y_train)}
ClientXY = Dict[int, Tuple[np.ndarray, np.ndarray]]

def _order_one_station(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    timestamps: Optional[Iterable] = None,
    timestamp_col_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Ensure (X, Y) are sorted by timestamp for one station."""
    if timestamps is not None:
        ts = np.asarray(list(timestamps))
        order = np.argsort(ts, kind="mergesort")
        return X[order], Y[order]
    if timestamp_col_idx is not None:
        order = np.argsort(X[:, timestamp_col_idx], kind="mergesort")
        return X[order], Y[order]
    return X, Y


class PerStationTumblingCoordinator:
    """
    Multi-station tumbling-window streamer with **per-station pull** API.

    - One background thread per station:
        * every `tick_interval` seconds: add up to `arrival_rate` samples
        * every `tumbling_time` seconds: emit ENTIRE buffer to that station's queue
    - Consumer pulls windows per-station with `get(sid, timeout=...)`.
    - Producer finishes after exactly one pass over each station's data.
    """

    DONE = object()

    def __init__(
        self,
        client_xy: ClientXY,
        *,
        station_ids: Optional[Iterable[int]] = None,
        arrival_rate: int,
        tick_interval: float,
        tumbling_time: float,
        timestamps_by_station: Optional[Dict[int, Iterable]] = None,
        timestamp_col_idx: Optional[int] = None,
        emit_empty_windows: bool = False,
        emit_final_partial: bool = True,
        queue_size_per_station: int = 16
    ):
        if arrival_rate <= 0:
            raise ValueError("arrival_rate must be >= 1")
        if tick_interval <= 0 or tumbling_time <= 0:
            raise ValueError("tick_interval and tumbling_time must be > 0")

        self.client_xy = client_xy
        self.stations: List[int] = (
            list(station_ids) if station_ids is not None else list(client_xy.keys())
        )
        self.arrival_rate = arrival_rate
        self.tick_interval = float(tick_interval)
        self.tumbling_time = float(tumbling_time)
        self.timestamps_by_station = timestamps_by_station or {}
        self.timestamp_col_idx = timestamp_col_idx
        self.emit_empty_windows = emit_empty_windows
        self.emit_final_partial = emit_final_partial

        # One queue per station
        self.queues: Dict[int, Queue] = {
            sid: Queue(maxsize=queue_size_per_station) for sid in self.stations
        }
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []

    # --------------------- public API ---------------------

    def start(self):
        if self._threads:
            return
        self._stop.clear()
        for sid in self.stations:
            if sid not in self.client_xy:
                raise KeyError(f"station_id {sid} not in client data")
            t = threading.Thread(target=self._station_worker, args=(sid,), daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=5.0)

    def get(self, sid: int, timeout: Optional[float] = None):
        """
        Pull next tumbling window for `sid`.
        Returns:
            (X_chunk, Y_chunk)
        Raises:
            queue.Empty if timeout and no item
            StopIteration if this station has finished (DONE seen)
        """
        q = self.queues[sid]
        item = q.get(timeout=timeout) if timeout is not None else q.get()
        if item is self.DONE:
            # Put it back so future gets also raise StopIteration consistently
            try:
                q.put_nowait(self.DONE)
            except:
                pass
            raise StopIteration(f"station {sid} finished")
        return item  # (X_chunk, Y_chunk)

    def get_nowait(self, sid: int):
        """Non-blocking fetch; returns None if no window ready yet; raises StopIteration if finished."""
        q = self.queues[sid]
        try:
            item = q.get_nowait()
        except Empty:
            return None
        if item is self.DONE:
            # keep DONE available
            try:
                q.put_nowait(self.DONE)
            except:
                pass
            raise StopIteration(f"station {sid} finished")
        return item

    def finished(self, sid: int) -> bool:
        """Check if a station is done (best-effort, non-blocking)."""
        q = self.queues[sid]
        try:
            item = q.get_nowait()
        except Empty:
            return False
        if item is self.DONE:
            # keep DONE available
            try:
                q.put_nowait(self.DONE)
            except:
                pass
            return True
        # put back normal item for later consumption
        q.put(item)
        return False

    # --------------------- internal worker ---------------------

    def _station_worker(self, sid: int):
        try:
            X, Y = self.client_xy[sid]
            if len(X) != len(Y):
                raise ValueError(f"station {sid}: X and Y length mismatch")

            X, Y = _order_one_station(
                X, Y,
                timestamps=self.timestamps_by_station.get(sid),
                timestamp_col_idx=self.timestamp_col_idx
            )

            N = len(X)
            cursor = 0
            bufX: List[np.ndarray] = []
            bufY: List[np.ndarray] = []
            window_start = time.time()
            next_tick = window_start

            q = self.queues[sid]

            while not self._stop.is_set():
                now = time.time()

                # Arrival tick: add up to arrival_rate samples
                if now >= next_tick and cursor < N:
                    a = cursor
                    b = min(a + self.arrival_rate, N)
                    if b > a:
                        bufX.append(X[a:b])
                        bufY.append(Y[a:b])
                        cursor = b
                    next_tick += self.tick_interval

                # Tumbling boundary: emit and clear buffer
                if (now - window_start) >= self.tumbling_time:
                    if self.emit_empty_windows or bufX:
                        X_chunk = (np.concatenate(bufX, axis=0) if len(bufX) > 1
                                   else (bufX[0] if bufX else X[:0]))
                        Y_chunk = (np.concatenate(bufY, axis=0) if len(bufY) > 1
                                   else (bufY[0] if bufY else Y[:0]))
                        if X_chunk.shape[0] > 0 or self.emit_empty_windows:
                            q.put((X_chunk, Y_chunk))
                        bufX.clear()
                        bufY.clear()
                    window_start = now

                # End condition
                if cursor >= N:
                    if self.emit_final_partial and bufX and not self._stop.is_set():
                        X_chunk = (np.concatenate(bufX, axis=0) if len(bufX) > 1 else bufX[0])
                        Y_chunk = (np.concatenate(bufY, axis=0) if len(bufY) > 1 else bufY[0])
                        if X_chunk.shape[0] > 0:
                            q.put((X_chunk, Y_chunk))
                    break

                time.sleep(min(0.01, self.tick_interval / 10))
        finally:
            # notify station completion
            self.queues[sid].put(self.DONE)
