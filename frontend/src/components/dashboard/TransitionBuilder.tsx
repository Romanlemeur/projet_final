"use client";

import { useState, type DragEvent } from "react";
import { motion } from "framer-motion";
import { mockTracks, type Track } from "@/lib/dashboard-data";

function GripIcon() {
	return (
		<svg className="w-4 h-4 text-white/20 shrink-0" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
			<circle cx="9" cy="6" r="1.5" />
			<circle cx="15" cy="6" r="1.5" />
			<circle cx="9" cy="12" r="1.5" />
			<circle cx="15" cy="12" r="1.5" />
			<circle cx="9" cy="18" r="1.5" />
			<circle cx="15" cy="18" r="1.5" />
		</svg>
	);
}

function SparklesIcon() {
	return (
		<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<path d="M12 2l2.09 6.26L20 10l-5.91 1.74L12 18l-2.09-6.26L4 10l5.91-1.74L12 2z" />
		</svg>
	);
}

function TrackPanel({
	label,
	selected,
	onDrop,
	onClear,
	dragOver,
	onDragOver,
	onDragLeave,
	onDragStart,
	exclude,
}: {
	label: string;
	selected: Track | null;
	onDrop: (e: DragEvent) => void;
	onClear: () => void;
	dragOver: boolean;
	onDragOver: (e: DragEvent) => void;
	onDragLeave: () => void;
	onDragStart: (e: DragEvent, track: Track) => void;
	exclude?: string;
}) {
	return (
		<div className="bento-card p-6 flex flex-col">
			<label className="block text-sm text-white/40 font-medium mb-4 uppercase tracking-wider">{label}</label>

			<div onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop} className={`relative rounded-xl border-2 border-dashed px-5 py-6 transition-colors mb-5 ${dragOver ? "border-green-500/60 bg-green-500/[0.06]" : selected ? "border-white/10 bg-white/[0.02]" : "border-white/[0.08] bg-transparent"}`}>
				{selected ? (
					<div className="flex items-center gap-4">
						<div className="w-12 h-12 rounded-lg shrink-0" style={{ backgroundColor: selected.color }} />
						<div className="flex-1 min-w-0">
							<p className="text-base font-semibold text-white truncate">{selected.title}</p>
							<p className="text-sm text-white/40">
								{selected.bpm} BPM · {selected.key}
							</p>
						</div>
						<button type="button" onClick={onClear} className="text-white/20 hover:text-white/50 transition-colors rounded p-1.5" aria-label={`Remove ${selected.title}`}>
							<svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
								<line x1="18" y1="6" x2="6" y2="18" />
								<line x1="6" y1="6" x2="18" y2="18" />
							</svg>
						</button>
					</div>
				) : (
					<p className="text-base text-white/25 text-center">Drag a track here</p>
				)}
			</div>

			<div className="max-h-56 overflow-y-auto space-y-2 pr-1">
				{mockTracks
					.filter((t) => t.id !== exclude)
					.map((track) => {
						const isSelected = selected?.id === track.id;
						return (
							<div
								key={track.id}
								draggable={!isSelected}
								onDragStart={(e) => onDragStart(e, track)}
								className={`flex items-center gap-3 rounded-lg px-3 py-3 border transition-colors select-none ${isSelected ? "border-green-500/20 bg-green-500/[0.04] opacity-40 cursor-default" : "border-white/[0.06] bg-white/[0.03] hover:border-white/10 cursor-grab active:cursor-grabbing"}`}
							>
								{!isSelected && <GripIcon />}
								<div className="w-8 h-8 rounded shrink-0" style={{ backgroundColor: track.color }} />
								<p className="text-sm font-medium text-white truncate flex-1">{track.title}</p>
								<span className="text-xs text-white/30 font-mono shrink-0">{track.bpm} BPM</span>
							</div>
						);
					})}
			</div>
		</div>
	);
}

export default function TransitionBuilder() {
	const [fromTrack, setFromTrack] = useState<Track | null>(null);
	const [toTrack, setToTrack] = useState<Track | null>(null);
	const [duration, setDuration] = useState(8);
	const [dragOverFrom, setDragOverFrom] = useState(false);
	const [dragOverTo, setDragOverTo] = useState(false);

	const canGenerate = fromTrack && toTrack;

	function handleDragStart(e: DragEvent, track: Track) {
		e.dataTransfer.setData("text/plain", track.id);
		e.dataTransfer.effectAllowed = "copy";
	}

	function handleDrop(e: DragEvent, target: "from" | "to") {
		e.preventDefault();
		const trackId = e.dataTransfer.getData("text/plain");
		const track = mockTracks.find((t) => t.id === trackId);
		if (!track) return;

		if (target === "from") {
			if (toTrack?.id === track.id) setToTrack(null);
			setFromTrack(track);
			setDragOverFrom(false);
		} else {
			if (fromTrack?.id === track.id) setFromTrack(null);
			setToTrack(track);
			setDragOverTo(false);
		}
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		e.dataTransfer.dropEffect = "copy";
	}

	return (
		<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="space-y-6">
			<div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
				<TrackPanel
					label="From Track"
					selected={fromTrack}
					onDrop={(e) => handleDrop(e, "from")}
					onClear={() => setFromTrack(null)}
					dragOver={dragOverFrom}
					onDragOver={(e) => {
						handleDragOver(e);
						setDragOverFrom(true);
					}}
					onDragLeave={() => setDragOverFrom(false)}
					onDragStart={handleDragStart}
					exclude={toTrack?.id}
				/>
				<TrackPanel
					label="To Track"
					selected={toTrack}
					onDrop={(e) => handleDrop(e, "to")}
					onClear={() => setToTrack(null)}
					dragOver={dragOverTo}
					onDragOver={(e) => {
						handleDragOver(e);
						setDragOverTo(true);
					}}
					onDragLeave={() => setDragOverTo(false)}
					onDragStart={handleDragStart}
					exclude={fromTrack?.id}
				/>
			</div>

			<button type="button" disabled={!canGenerate} className={`w-full flex items-center justify-center gap-2.5 py-4 rounded-xl font-semibold text-base transition-all ${canGenerate ? "bg-green-500 text-black hover:bg-green-400 hover:scale-[1.01] shadow-[0_0_40px_-10px_rgba(29,185,84,0.5)]" : "bg-white/5 text-white/20 cursor-not-allowed"}`}>
				<SparklesIcon />
				Generate Transition
			</button>
		</motion.div>
	);
}
