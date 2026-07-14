#!/usr/bin/env bash
set -euo pipefail

ADB="${ADB:-${ANDROID_HOME:-$HOME/Android/Sdk}/platform-tools/adb}"
PACKAGE="${OA_ANDROID_PACKAGE:-com.oa.mobilelab.debug}"
ACTIVITY="$PACKAGE/com.oa.mobilelab.MainActivity"
STEPS="${OA_NLP_STEPS:-300}"
BATCH="${OA_NLP_BATCH:-64}"
TIMEOUT_SECONDS="${OA_NLP_TIMEOUT_SECONDS:-1800}"

if (($# == 0)); then
	architectures=(rnn gru transformer moe mamba3)
	tokenizers=(byte)
elif [[ "${1}" == "all" ]]; then
	architectures=(rnn gru transformer moe mamba3)
	tokenizers=(byte bpe char)
else
	architectures=("${1:?architecture required}")
	tokenizers=("${2:?tokenizer required}")
fi

"$ADB" get-state >/dev/null

for tokenizer in "${tokenizers[@]}"; do
	for architecture in "${architectures[@]}"; do
		run_id="$tokenizer-$architecture"
		report="files/reports/$run_id-training.txt"
		echo "=== $tokenizer / $architecture · $STEPS steps · batch $BATCH ==="
		"$ADB" shell "run-as $PACKAGE sh -c 'rm -f \"$report\" files/checkpoints/$run_id*.oam'"
		"$ADB" shell am force-stop "$PACKAGE"
		"$ADB" shell am start -S -W --activity-clear-task \
			-n "$ACTIVITY" \
			--es architecture "$architecture" \
			--es tokenizer "$tokenizer" \
			--ez train true --ei steps "$STEPS" --ei batch "$BATCH" >/dev/null

		deadline=$((SECONDS + TIMEOUT_SECONDS))
		while ! "$ADB" shell "run-as $PACKAGE test -f '$report'"; do
			if ((SECONDS >= deadline)); then
				echo "timeout waiting for $run_id" >&2
				exit 1
			fi
			sleep 2
		done

		result=$("$ADB" shell "run-as $PACKAGE cat '$report'" | tr -d '\r')
		printf '%s\n' "$result"
		grep -Fq "Steps: $STEPS/$STEPS" <<<"$result"
		grep -Fq "Cancelled: no" <<<"$result"
		grep -Fq "CheckpointRoundTrip: PASS" <<<"$result"
		grep -Fq "GenerationQuality: PASS" <<<"$result"
		grep -Fq "Prompt: 'to be'" <<<"$result"
		grep -Fq "Generated: '" <<<"$result"
	done
done

echo "All Android NLP suite routes passed."
