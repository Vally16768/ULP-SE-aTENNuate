from __future__ import annotations

from pathlib import Path


class ProcessLock:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.handle = None

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        try:
            try:
                import msvcrt

                self.handle.seek(0)
                self.handle.write(b"0")
                self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            except ImportError:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.release()
            return False
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(str(__import__("os").getpid()).encode("ascii"))
        self.handle.flush()
        return True

    def release(self) -> None:
        if self.handle is None:
            return
        try:
            try:
                import msvcrt

                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            except ImportError:
                import fcntl

                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        self.handle.close()
        self.handle = None

    def __enter__(self) -> "ProcessLock":
        if not self.acquire():
            raise RuntimeError(f"Unable to acquire process lock: {self.path}")
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
