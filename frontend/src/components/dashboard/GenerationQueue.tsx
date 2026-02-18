"use client";

import { motion } from "framer-motion";
import { mockQueue } from "@/lib/dashboard-data";

function CheckIcon() {
	return (
		<svg className="w-3.5 h-3.5 text-green-500" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<polyline points="20 6 9 17 4 12" />
		</svg>
	);
}

function SpinnerIcon() {
	return (
		<svg className="w-3.5 h-3.5 text-green-500 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
			<circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="31.4 31.4" strokeLinecap="round" />
		</svg>
	);
}

function ClockIcon() {
	return (
		<svg className="w-3.5 h-3.5 text-white/30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<circle cx="12" cy="12" r="10" />
			<polyline points="12 6 12 12 16 14" />
		</svg>
	);
}

const STATUS_CONFIG = {
	completed: { icon: <CheckIcon />, label: "Completed", color: "text-green-500" },
	processing: { icon: <SpinnerIcon />, label: "Processing…", color: "text-green-500" },
	pending: { icon: <ClockIcon />, label: "Queued", color: "text-white/30" },
} as const;

function Waveform({ status }: { status: "pending" | "processing" | "completed" }) {
	const barCount = 70;
	const bars = Array.from({ length: barCount }, (_, i) => {
		const h = ((Math.sin(i * 0.7) + 1) / 2) * 0.6 + 0.2;
		return h;
	});

	const zoneStart = Math.floor(barCount * 0.35);
	const zoneEnd = Math.floor(barCount * 0.65);

	return (
		<div className="flex items-end gap-[2px] h-12">
			{bars.map((h, i) => {
				const inZone = i >= zoneStart && i <= zoneEnd;
				let color: string;
				if (status === "completed") {
					color = inZone ? "bg-green-500" : "bg-white/15";
				} else if (status === "processing") {
					color = inZone ? "bg-green-500/50" : "bg-white/10";
				} else {
					color = "bg-white/8";
				}

				return <div key={i} className={`w-[3px] rounded-full ${color}`} style={{ height: `${h * 100}%` }} />;
			})}
		</div>
	);
}

export default function GenerationQueue() {
	if (mockQueue.length === 0) {
		return (
			<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="flex flex-col items-center justify-center py-16 text-center">
				<p className="text-white/30 text-sm">No transitions in the queue yet.</p>
				<p className="text-white/20 text-xs mt-1">Use the Transition Builder to generate your first one.</p>
			</motion.div>
		);
	}

	return (
		<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="space-y-4">
			{mockQueue.map((item) => {
				const cfg = STATUS_CONFIG[item.status];
				return (
					<div key={item.id} className="bento-card p-5 sm:p-7 space-y-4">
						<div className="flex items-center justify-between gap-5">
							<div className="flex items-center gap-4 min-w-0">
								<div className="flex items-center gap-3 min-w-0">
									<div className="w-10 h-10 rounded-lg shrink-0" style={{ backgroundColor: item.fromTrack.color }} />
									<span className="text-base font-medium text-white truncate">{item.fromTrack.title}</span>
								</div>

								<span className="text-white/20 text-sm shrink-0">→</span>

								<div className="flex items-center gap-3 min-w-0">
									<div className="w-10 h-10 rounded-lg shrink-0" style={{ backgroundColor: item.toTrack.color }} />
									<span className="text-base font-medium text-white truncate">{item.toTrack.title}</span>
								</div>
							</div>

							<div className={`flex items-center gap-2 shrink-0 ${cfg.color}`}>
								{cfg.icon}
								<span className="text-sm font-medium">{cfg.label}</span>
							</div>
						</div>

						<Waveform status={item.status} />

						<div className="flex items-center gap-5 text-sm text-white/30">
							<span>{item.transitionDuration}s transition</span>
							<span>
								{item.fromBpm} → {item.toBpm} BPM
							</span>
						</div>
					</div>
				);
			})}
		</motion.div>
	);
}
