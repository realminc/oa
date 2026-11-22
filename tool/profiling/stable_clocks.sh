#!/usr/bin/env bash
set -euo pipefail

gpu_mhz=1000
cpu_khz=2600000
power_profile=performance
cpu_governor=performance

usage() {
	printf 'usage: %s [--gpu-mhz MHz] [--cpu-khz kHz] -- command [args...]\n' "$0" >&2
}

while (( $# )); do
	case "$1" in
		--gpu-mhz)
			(( $# >= 2 )) || { usage; exit 2; }
			gpu_mhz=$2
			shift 2
			;;
		--cpu-khz|--cpu-max-khz)
			(( $# >= 2 )) || { usage; exit 2; }
			cpu_khz=$2
			shift 2
			;;
		--)
			shift
			break
			;;
		*)
			usage
			exit 2
			;;
	esac
done

(( $# )) || { usage; exit 2; }
[[ $gpu_mhz =~ ^[0-9]+$ && $cpu_khz =~ ^[0-9]+$ ]] || {
	printf 'stable_clocks: frequencies must be positive integers\n' >&2
	exit 2
}

no_turbo=/sys/devices/system/cpu/intel_pstate/no_turbo
[[ -r $no_turbo && $(<"$no_turbo") == 0 ]] || {
	printf 'stable_clocks: Intel Turbo must be enabled (intel_pstate/no_turbo=0)\n' >&2
	exit 1
}

shopt -s nullglob
cpu_roots=(/sys/devices/system/cpu/cpufreq/policy*)
(( ${#cpu_roots[@]} )) || {
	printf 'stable_clocks: no CPU frequency policies found\n' >&2
	exit 1
}
for root in "${cpu_roots[@]}"; do
	hardware_min=$(<"$root/cpuinfo_min_freq")
	hardware_max=$(<"$root/cpuinfo_max_freq")
	(( cpu_khz >= hardware_min && cpu_khz <= hardware_max )) || {
		printf 'stable_clocks: requested %s kHz is outside %s range %s-%s kHz\n' \
			"$cpu_khz" "$root" "$hardware_min" "$hardware_max" >&2
		exit 1
	}
done

gpu_roots=(/sys/class/drm/card[0-9]*/device/tile*/gt*/freq*)
if (( ${#gpu_roots[@]} == 1 )); then
	gpu_root=${gpu_roots[0]}
	gpu_min=$gpu_root/min_freq
	gpu_max=$gpu_root/max_freq
	gpu_rpn=$(<"$gpu_root/rpn_freq")
	gpu_rp0=$(<"$gpu_root/rp0_freq")
elif (( ${#gpu_roots[@]} == 0 )); then
	# i915 exposes the GT RPS frequency bounds under gt/gt*/rps_* instead.
	gpu_roots=(/sys/class/drm/card[0-9]*/gt/gt*)
	if (( ${#gpu_roots[@]} == 1 )) \
		&& [[ -r ${gpu_roots[0]}/rps_min_freq_mhz \
			&& -r ${gpu_roots[0]}/rps_max_freq_mhz \
			&& -r ${gpu_roots[0]}/rps_RP0_freq_mhz \
			&& -r ${gpu_roots[0]}/rps_RPn_freq_mhz ]]; then
		gpu_root=${gpu_roots[0]}
		gpu_min=$gpu_root/rps_min_freq_mhz
		gpu_max=$gpu_root/rps_max_freq_mhz
		gpu_rpn=$(<"$gpu_root/rps_RPn_freq_mhz")
		gpu_rp0=$(<"$gpu_root/rps_RP0_freq_mhz")
	else
		printf 'stable_clocks: no Xe tile*/gt*/freq* or i915 gt/gt*/rps_* frequency engine found\n' >&2
		exit 1
	fi
else
	printf 'stable_clocks: expected exactly one Xe frequency engine, found %d\n' \
		"${#gpu_roots[@]}" >&2
	exit 1
fi
(( gpu_mhz >= gpu_rpn && gpu_mhz <= gpu_rp0 )) || {
	printf 'stable_clocks: requested %s MHz is outside Xe range %s-%s MHz\n' \
		"$gpu_mhz" "$gpu_rpn" "$gpu_rp0" >&2
	exit 1
}

original_gpu_min=$(<"$gpu_min")
original_gpu_max=$(<"$gpu_max")
original_power_profile=
if command -v powerprofilesctl >/dev/null; then
	original_power_profile=$(powerprofilesctl get)
fi
power_profile_changed=0
cpu_clocks_touched=0
gpu_clocks_touched=0
declare -a original_cpu_min=()
declare -a original_cpu_max=()
declare -a original_cpu_governor=()
for root in "${cpu_roots[@]}"; do
	original_cpu_min+=("$(<"$root/scaling_min_freq")")
	original_cpu_max+=("$(<"$root/scaling_max_freq")")
	original_cpu_governor+=("$(<"$root/scaling_governor")")
done

# Privileged writes are queued and flushed as ONE pkexec invocation per
# phase — a single authentication covers the whole pin or restore.
declare -a pending_paths=() pending_values=()

queue_write() {
	pending_paths+=("$1")
	pending_values+=("$2")
}

flush_writes() {
	(( ${#pending_paths[@]} )) || return 0
	local index needs_pkexec=0 script= line
	for index in "${!pending_paths[@]}"; do
		[[ -w ${pending_paths[$index]} ]] || needs_pkexec=1
		printf -v line 'printf "%%s\\n" %q > %q\n' 			"${pending_values[$index]}" "${pending_paths[$index]}"
		script+=$line
	done
	pending_paths=()
	pending_values=()
	if (( needs_pkexec )); then
		command -v pkexec >/dev/null || {
			printf 'stable_clocks: pkexec is required for privileged sysfs writes\n' >&2
			return 1
		}
		pkexec bash -c "$script"
	else
		bash -c "$script"
	fi
}

write_range() {
	local minimum_path=$1
	local maximum_path=$2
	local minimum=$3
	local maximum=$4
	local current_min
	current_min=$(<"$minimum_path")
	if (( current_min > maximum )); then
		queue_write "$minimum_path" "$minimum"
		queue_write "$maximum_path" "$maximum"
	else
		queue_write "$maximum_path" "$maximum"
		queue_write "$minimum_path" "$minimum"
	fi
}

restore_clocks() {
	local status=$?
	trap - EXIT INT TERM HUP
	if (( cpu_clocks_touched )); then
		for index in "${!cpu_roots[@]}"; do
			write_range "${cpu_roots[$index]}/scaling_min_freq" \
				"${cpu_roots[$index]}/scaling_max_freq" \
				"${original_cpu_min[$index]}" "${original_cpu_max[$index]}"
			queue_write "${cpu_roots[$index]}/scaling_governor" \
				"${original_cpu_governor[$index]}"
		done
	fi
	if (( gpu_clocks_touched )); then
		write_range "$gpu_min" "$gpu_max" \
			"$original_gpu_min" "$original_gpu_max"
	fi
	flush_writes || status=1
	if (( power_profile_changed )); then
		powerprofilesctl set "$original_power_profile" || status=1
	fi
	printf 'OARS_BENCH_CLOCKS restored power_profile=%s cpu_policies=%s gpu_min_mhz=%s gpu_max_mhz=%s\n' \
		"$original_power_profile" "${#cpu_roots[@]}" "$original_gpu_min" "$original_gpu_max" >&2
	exit "$status"
}
trap restore_clocks EXIT INT TERM HUP

if [[ -n $original_power_profile ]]; then
	power_profile_changed=1
	powerprofilesctl set "$power_profile"
	[[ $(powerprofilesctl get) == "$power_profile" ]] || {
		printf 'stable_clocks: performance power profile request did not stick\n' >&2
		exit 1
	}
	performance_degradation=$(busctl get-property net.hadess.PowerProfiles \
		/net/hadess/PowerProfiles net.hadess.PowerProfiles PerformanceDegraded \
		2>/dev/null) || {
		printf 'stable_clocks: performance degradation state is unavailable\n' >&2
		exit 1
	}
	if [[ $performance_degradation != 's ""' ]]; then
		printf 'stable_clocks: performance power profile is degraded: %s\n' \
			"$performance_degradation" >&2
		exit 1
	fi
else
	printf 'stable_clocks: powerprofilesctl not installed; pinning clocks without a power profile\n' >&2
fi

cpu_clocks_touched=1
gpu_clocks_touched=1
for root in "${cpu_roots[@]}"; do
	write_range "$root/scaling_min_freq" "$root/scaling_max_freq" "$cpu_khz" "$cpu_khz"
	queue_write "$root/scaling_governor" "$cpu_governor"
done
write_range "$gpu_min" "$gpu_max" "$gpu_mhz" "$gpu_mhz"
# One privileged invocation covers every CPU policy and the GT bounds.
flush_writes
for root in "${cpu_roots[@]}"; do
	[[ $(<"$root/scaling_min_freq") == "$cpu_khz" \
		&& $(<"$root/scaling_max_freq") == "$cpu_khz" \
		&& $(<"$root/scaling_governor") == "$cpu_governor" ]] || {
		printf 'stable_clocks: fixed CPU policy request did not stick for %s\n' "$root" >&2
		exit 1
	}
done

[[ $(<"$gpu_min") == "$gpu_mhz" && $(<"$gpu_max") == "$gpu_mhz" ]] || {
	printf 'stable_clocks: Xe fixed-frequency request did not stick\n' >&2
	exit 1
}

export OA_BENCH_CPU_FIXED_KHZ=$cpu_khz
export OA_BENCH_CPU_MAX_KHZ=$cpu_khz
export OA_BENCH_GPU_FIXED_MHZ=$gpu_mhz
export OA_BENCH_CPU_GOVERNOR=$cpu_governor
export OA_BENCH_POWER_PROFILE=$power_profile
printf 'OARS_BENCH_CLOCKS power_profile=%s cpu_governor=%s cpu_turbo=enabled cpu_fixed_khz=%s gpu_fixed_mhz=%s\n' \
	"$power_profile" "$cpu_governor" "$cpu_khz" "$gpu_mhz" >&2
"$@"
