import sys
import datetime

class Tee:
        """
        Write to both stdout and a file
        """
        def __init__(self, file_path: str):
                self.file_path = file_path
                self.f = open(file_path, "a", encoding="utf-8")
                self.stdout = sys.stdout

        def write(self, out):
                self.stdout.write(out)
                self.f.write(out)

        def flush(self):
                self.stdout.flush()
                self.f.flush()

        def close(self):
                sys.stdout = self.stdout
                self.f.close()

        def __enter__(self):
                sys.stdout = self
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.write(f"\n\n[START] {now}\n\n")
                return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                self.write(f"\n[END] {now}\n\n\n")
                self.close()
