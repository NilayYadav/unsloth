from pathlib import Path
import re
import subprocess


def build_environments(root, out):
    sources = {
        "before": subprocess.check_output(
            ["git", "show", "a1b398e7d6024b43d2db4632327599c15f401b0c:studio/src-tauri/src/update.rs"],
            cwd=root, text=True, encoding="utf-8",
        ),
        "after": (root / "studio/src-tauri/src/update.rs").read_text(encoding="utf-8"),
    }
    prefix = "use std::process::Command;\nmod preflight { pub fn expected_backend_version() -> &'static str { \"test\" } }\n"
    main = '''
fn main() {
    let mut command = Command::new("unused");
    configure_tauri_update_environment(&mut command);
    for (key, value) in command.get_envs() {
        if let Some(value) = value {
            println!("{}={}", key.to_string_lossy(), value.to_string_lossy());
        }
    }
}
'''
    environments = {}
    for side, source in sources.items():
        function = re.search(r"fn configure_tauri_update_environment\(.*?^\}", source, re.M | re.S).group()
        rust = out / f"environment-{side}.rs"
        rust.write_text(prefix + function + main, encoding="utf-8")
        executable = out / f"environment-{side}.exe"
        subprocess.run(["rustc", "--edition=2021", "--crate-name", "update_environment", str(rust), "-o", str(executable)], check=True)
        result = subprocess.check_output([str(executable)], text=True, encoding="utf-8")
        environments[side] = dict(line.split("=", 1) for line in result.splitlines())
    assert "UNSLOTH_PROGRESS_PERCENT_STEP" not in environments["before"]
    assert environments["after"]["UNSLOTH_PROGRESS_PERCENT_STEP"] == "5"
    test = re.search(
        r"    #\[test\]\n    fn tauri_backend_update_skips_the_web_frontend_build\(\).*?^    \}",
        sources["after"], re.M | re.S,
    ).group()
    rust = out / "environment-test.rs"
    rust.write_text(prefix + function + test, encoding="utf-8")
    executable = out / "environment-test.exe"
    subprocess.run(["rustc", "--edition=2021", "--crate-name", "update_environment_test", "--test", str(rust), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
    return environments
