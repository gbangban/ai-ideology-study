import os
import sys

write_token = os.environ.get("TRACKIO_WRITE_TOKEN")
if write_token:
    import trackio.server
    trackio.server.write_token = write_token

from trackio.cli import main
sys.argv = ["trackio", "show", "--host", "0.0.0.0"]
main()
