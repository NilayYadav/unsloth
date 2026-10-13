#!/usr/bin/env bash
set -uo pipefail

REPO="$GITHUB_WORKSPACE"
H="$REPO/.github/pr13660"
OUT="$RUNNER_TEMP/out"
PROJ="$RUNNER_TEMP/proj"
IMG=unsloth/unsloth:latest
export JUPYTER_PASSWORD=pr13660-local-only
mkdir -p "$OUT" "$PROJ" "$HOME/.cache/huggingface"
echo "host file" > "$PROJ/host-file.txt"
cd "$PROJ"

SAVE='from transformers import GPT2Config, GPT2LMHeadModel; GPT2LMHeadModel(GPT2Config(n_layer=1, n_embd=32, n_head=2, vocab_size=128)).save_pretrained("lora_model"); print("saved lora_model")'

cid() { docker ps -q --filter "ancestor=$1" | head -1; }

wait_cid() {
    local c="" i
    for i in $(seq 1 120); do c="$(cid "$1")"; [ -n "$c" ] && break; sleep 2; done
    printf '%s' "$c"
}

runsh_up() {
    local log="$1"; shift
    env "$@" UNSLOTH_GPUS=none UNSLOTH_ALLOW_CPU=1 UNSLOTH_PORTS="-p 8000:8000 -p 8888:8888" \
        bash "$REPO/docker/run.sh" > "$OUT/$log" 2>&1 &
    RUNSH_PID=$!
}

runsh_down() {
    local c="$1"
    docker stop "$c" > /dev/null
    wait "$RUNSH_PID" 2> /dev/null || true
    for _ in $(seq 1 30); do docker ps -aq --no-trunc | grep -q "^$c" || break; sleep 1; done
    docker ps -aq --no-trunc | grep -q "^$c" && echo "container $c still exists" || echo "container ${c:0:12} removed"
}

user_session() {
    local c="$1"
    python3 "$H/jl.py" wait
    NB="$(python3 "$H/jl.py" edit | tail -1)"
    docker exec -w "/workspace/$(dirname "$NB")" "$c" python -c "$SAVE"
    echo "before removal: $(python3 "$H/jl.py" check "$NB")"
}

verdict() {
    python3 -c 'import json,sys; d=json.loads(sys.argv[1]); print("PERSISTED" if d["edit"] and "model.safetensors" in d["lora_model"] else "LOST")' "$1"
}

echo "::group::documented run.sh, boot 1"
runsh_up runsh-boot1.log
C1="$(wait_cid "$IMG")"
docker inspect -f '{{range .Mounts}}{{.Type}} {{.Name}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' "$C1" | sort | tee "$OUT/runsh-mounts.txt"
user_session "$C1"
runsh_down "$C1"
echo "::endgroup::"

echo "::group::documented run.sh, boot 2"
runsh_up runsh-boot2.log
C2="$(wait_cid "$IMG")"
python3 "$H/jl.py" wait
R1="$(python3 "$H/jl.py" check "$NB")"
echo "after docker rm + run: $R1"
docker exec "$C2" cat /workspace/host/host-file.txt
docker exec "$C2" touch /workspace/.cache/huggingface/pr13660-probe
ls "$HOME/.cache/huggingface/pr13660-probe" && echo "hf cache bind still reaches the host"
for _ in $(seq 1 60); do grep -qi "earlier boot\|generated" "$OUT/runsh-boot2.log" && break; sleep 5; done
grep -i "earlier boot" "$OUT/runsh-boot2.log" | head -2
runsh_down "$C2"
echo "::endgroup::"

echo "::group::non-root uid on the same volume"
docker run --rm --user 1000:1000 -v unsloth-workspace:/workspace --entrypoint bash "$IMG" \
    -c 'touch /workspace/pr13660-uid1000 && echo "uid 1000 can write /workspace"; ls -ld /workspace' 2>&1 || true
echo "::endgroup::"

echo "::group::image update on the existing volume"
printf 'FROM %s\nRUN echo "# next release" >> /workspace/smoke_test.py\n' "$IMG" | docker build -q -t unsloth/unsloth:pr13660-next - > /dev/null
runsh_up runsh-update.log UNSLOTH_IMAGE=unsloth/unsloth:pr13660-next
C3="$(wait_cid unsloth/unsloth:pr13660-next)"
python3 "$H/jl.py" wait
R_UPD="$(python3 "$H/jl.py" check "$NB")"
echo "after image update: $R_UPD"
IN_CTR="$(docker exec "$C3" sha256sum /workspace/smoke_test.py | cut -c1-12)"
IN_IMG="$(docker run --rm --entrypoint sha256sum unsloth/unsloth:pr13660-next /workspace/smoke_test.py | cut -c1-12)"
echo "smoke_test.py in container $IN_CTR, in new image $IN_IMG"
docker exec "$C3" ls -1 /workspace
runsh_down "$C3"
echo "::endgroup::"

echo "::group::UNSLOTH_WORKSPACE_VOLUME empty"
runsh_up runsh-off.log UNSLOTH_WORKSPACE_VOLUME=
C4="$(wait_cid "$IMG")"
docker inspect -f '{{range .Mounts}}{{.Type}} {{.Name}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' "$C4" | sort | tee "$OUT/runsh-off-mounts.txt"
OFF_WS="$(grep -c ' -> /workspace$' "$OUT/runsh-off-mounts.txt" || true)"
runsh_down "$C4"
echo "::endgroup::"

docker volume ls
docker volume rm unsloth-workspace unsloth-studio > /dev/null 2>&1 || true

echo "::group::Docker Hub quick start, boot 1"
CMD="$(python3 - "$REPO/docker/DOCKERHUB.md" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
block = text.split("## Quick start", 1)[1].split("```bash", 1)[1].split("```", 1)[0]
lines = [l for l in block.splitlines() if l.strip() and not l.lstrip().startswith("#")]
cmd = " ".join(l.rstrip().rstrip("\\") for l in lines)
cmd = cmd.replace("--gpus all ", "").split("&&")[0].strip()
cmd = cmd.replace("docker run -d", "docker run -d -e JUPYTER_PASSWORD", 1)
print(re.sub(r"\s+", " ", cmd))
PY
)"
echo "$CMD"
eval "$CMD"
user_session unsloth
docker rm -f unsloth
echo "::endgroup::"

echo "::group::Docker Hub quick start, boot 2"
eval "$CMD"
python3 "$H/jl.py" wait
R2="$(python3 "$H/jl.py" check "$NB")"
echo "after docker rm -f + run: $R2"
docker inspect -f '{{range .Mounts}}{{.Type}} {{.Name}}{{.Source}} -> {{.Destination}}{{"\n"}}{{end}}' unsloth | sort
for _ in $(seq 1 60); do docker logs unsloth 2>&1 | grep -qi "studio" && docker logs unsloth 2>&1 | grep -qi "earlier boot\|generated" && break; sleep 5; done
docker logs unsloth 2>&1 | grep -i "earlier boot" | head -2
docker rm -f unsloth > /dev/null
echo "::endgroup::"

V1="$(verdict "$R1")"; V2="$(verdict "$R2")"; VU="$(verdict "$R_UPD")"
echo "RESULT run.sh docker rm cycle: $V1 $R1"
echo "RESULT Hub quick start docker rm -f cycle: $V2 $R2"
echo "RESULT image update: $VU smoke_test.py container=$IN_CTR image=$IN_IMG"
echo "RESULT UNSLOTH_WORKSPACE_VOLUME empty: $OFF_WS mount(s) on /workspace"
docker volume rm unsloth-workspace unsloth-studio > /dev/null 2>&1 || true

if [ "$V1" = PERSISTED ] && [ "$V2" = PERSISTED ] && [ "$OFF_WS" = 0 ]; then
    echo "PASS: notebook edit and saved model survive docker rm"
else
    echo "FAIL: notebook edit or saved model lost after docker rm (run.sh=$V1 hub=$V2)"
    exit 1
fi
