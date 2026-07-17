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

if "$ADB" shell "run-as $PACKAGE true" >/dev/null 2>&1; then
	report_transport="file"
else
	report_transport="logcat"
fi
echo "Report transport: $report_transport"

for tokenizer in "${tokenizers[@]}"; do
	for architecture in "${architectures[@]}"; do
		run_id="$tokenizer-$architecture"
		report="files/reports/$run_id-training.txt"
		echo "=== $tokenizer / $architecture · $STEPS steps · batch $BATCH ==="
		if [[ "$report_transport" == "file" ]]; then
			"$ADB" shell "run-as $PACKAGE sh -c 'rm -f \"$report\" files/checkpoints/$run_id*.oam'"
		else
			"$ADB" logcat -c
		fi
		"$ADB" shell am force-stop "$PACKAGE"
		"$ADB" shell am start -S -W --activity-clear-task \
			-n "$ACTIVITY" \
			--es architecture "$architecture" \
			--es tokenizer "$tokenizer" \
			--ez train true --ei steps "$STEPS" --ei batch "$BATCH" >/dev/null

		deadline=$((SECONDS + TIMEOUT_SECONDS))
		while true; do
			if [[ "$report_transport" == "file" ]]; then
				"$ADB" shell "run-as $PACKAGE test -f '$report'" && break
			else
				logs=$("$ADB" logcat -d -v raw -s OaMobileReport:I '*:S' 2>/dev/null)
				grep -Fq "OA_REPORT_END $run_id" <<<"$logs" && break
			fi
			if ((SECONDS >= deadline)); then
				echo "timeout waiting for $run_id" >&2
				exit 1
			fi
			sleep 2
		done

		if [[ "$report_transport" == "file" ]]; then
			result=$("$ADB" shell "run-as $PACKAGE cat '$report'" | tr -d '\r')
		else
			result=$(awk -v prefix="OA_REPORT $run_id " \
				'index($0, prefix) == 1 { print substr($0, length(prefix) + 1) }' \
				<<<"$logs")
		fi
		printf '%s\n' "$result"
		grep -Fq "Steps: $STEPS/$STEPS" <<<"$result"
		grep -Fq "Cancelled: no" <<<"$result"
		grep -Fq "CheckpointRoundTrip: PASS" <<<"$result"
		if ((STEPS >= 300)); then
			grep -Fq "GenerationQuality: PASS" <<<"$result"
			grep -Fq "Prompt: 'to be'" <<<"$result"
			grep -Fq "Generated: '" <<<"$result"
		else
			grep -Fq "GenerationQuality: not evaluated (<300 optimizer steps)" <<<"$result"
		fi
	done
done

echo "All Android NLP suite routes passed."
