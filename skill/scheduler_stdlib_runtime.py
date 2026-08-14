"""Standard-library dependency exports for runtime-bound compatibility wrappers."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time


def build_stdlib_runtime_exports(namespace: dict) -> dict:
    return {
        "argparse": argparse,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
        "re": re,
        "shlex": shlex,
        "signal": signal,
        "subprocess": subprocess,
        "sys": sys,
        "threading": threading,
        "time": time,
    }
