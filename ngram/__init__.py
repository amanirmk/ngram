import logging
import sys

import coloredlogs


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
coloredlogs.install(level="INFO")
