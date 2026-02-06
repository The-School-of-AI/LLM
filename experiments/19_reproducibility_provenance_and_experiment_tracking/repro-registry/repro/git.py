import subprocess

def git_info():
    def cmd(c):
        return subprocess.check_output(c, stderr=subprocess.DEVNULL).decode().strip()

    return {
        "repo": cmd(["git", "config", "--get", "remote.origin.url"]),
        "commit": cmd(["git", "rev-parse", "HEAD"]),
        "dirty": bool(cmd(["git", "status", "--porcelain"]))
    }
