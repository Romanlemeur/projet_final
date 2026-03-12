"use client";

import { useState, type DragEvent } from "react";
import { motion } from "framer-motion";
import { type Track, type QueueItem } from "@/lib/dashboard-data";

const API_URL = "http://localhost:8000";

const CAMELOT: Record<string, string> = {
	C_major: "8B",
	A_minor: "8A",
	G_major: "9B",
	E_minor: "9A",
	D_major: "10B",
	B_minor: "10A",
	A_major: "11B",
	"F#_minor": "11A",
	E_major: "12B",
	"C#_minor": "12A",
	B_major: "1B",
	"G#_minor": "1A",
	"F#_major": "2B",
	"D#_minor": "2A",
	"C#_major": "3B",
	"A#_minor": "3A",
	"G#_major": "4B",
	F_minor: "4A",
	"D#_major": "5B",
	C_minor: "5A",
	"A#_major": "6B",
	G_minor: "6A",
	F_major: "7B",
	D_minor: "7A",
	Db_major: "3B",
	Eb_major: "5B",
	Ab_major: "4B",
	Bb_major: "6B",
	Gb_major: "2B",
};

function getCamelot(key: string, mode: string): string {
	return CAMELOT[`${key}_${mode}`] ?? "8B";
}

function getCompatibility(t1: Track, t2: Track): { label: string; color: string; score: number; cam1: string; cam2: string } | null {
	if (!t1.key || !t1.mode || !t2.key || !t2.mode) return null;
	const cam1 = getCamelot(t1.key, t1.mode);
	const cam2 = getCamelot(t2.key, t2.mode);
	const num1 = parseInt(cam1),
		num2 = parseInt(cam2);
	const letter1 = cam1.slice(-1),
		letter2 = cam2.slice(-1);
	if (cam1 === cam2) return { label: "Perfect match", color: "#1DB954", score: 1.0, cam1, cam2 };
	if (num1 === num2 && letter1 !== letter2) return { label: "Relative key", color: "#1DB954", score: 0.95, cam1, cam2 };
	const dist = Math.min(Math.abs(num1 - num2), 12 - Math.abs(num1 - num2));
	if (dist === 1) return { label: "Adjacent key", color: "#1DB954", score: 0.88, cam1, cam2 };
	if (dist === 2) return { label: "Acceptable", color: "#f59e0b", score: 0.7, cam1, cam2 };
	if (dist <= 3) return { label: "Creates tension", color: "#ef4444", score: 0.5, cam1, cam2 };
	return { label: "Key clash", color: "#ef4444", score: 0.25, cam1, cam2 };
}

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
	tracks,
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
	tracks: Track[];
	exclude?: string;
}) {
	return (
		<div className="bento-card p-6 flex flex-col gap-4">
			<label className="text-xs text-white/40 font-medium uppercase tracking-wider">{label}</label>

			<div onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop} className={`rounded-xl border-2 border-dashed px-5 py-5 transition-colors ${dragOver ? "border-green-500/60 bg-green-500/[0.06]" : selected ? "border-white/10 bg-white/[0.02]" : "border-white/[0.08]"}`}>
				{selected ? (
					<div className="flex items-center gap-3">
						<div className="w-10 h-10 rounded-lg shrink-0 flex items-center justify-center text-black font-bold" style={{ backgroundColor: selected.color }}>
							{selected.title.charAt(0).toUpperCase()}
						</div>
						<div className="flex-1 min-w-0">
							<p className="text-sm font-semibold text-white truncate">{selected.title}</p>
							<p className="text-xs text-white/40">{selected.duration}</p>
						</div>
						<button type="button" onClick={onClear} className="text-white/20 hover:text-white/60 p-1.5 rounded" aria-label="Remove">
							<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
								<line x1="18" y1="6" x2="6" y2="18" />
								<line x1="6" y1="6" x2="18" y2="18" />
							</svg>
						</button>
					</div>
				) : (
					<p className="text-sm text-white/25 text-center py-1">Drag a track here</p>
				)}
			</div>

			<div className="max-h-52 overflow-y-auto space-y-1.5 pr-1">
				{tracks.filter((t) => t.id !== exclude).length === 0 ? (
					<p className="text-xs text-white/20 text-center py-4">No tracks yet - upload some in the Library tab.</p>
				) : (
					tracks
						.filter((t) => t.id !== exclude)
						.map((track) => {
							const isSelected = selected?.id === track.id;
							return (
								<div
									key={track.id}
									draggable={!isSelected}
									onDragStart={(e) => onDragStart(e, track)}
									className={`flex items-center gap-3 rounded-lg px-3 py-2.5 border transition-colors select-none ${isSelected ? "border-green-500/20 bg-green-500/[0.04] opacity-40 cursor-default" : "border-white/[0.06] bg-white/[0.03] hover:border-white/10 cursor-grab active:cursor-grabbing"}`}
								>
									{!isSelected && <GripIcon />}
									<div className="w-7 h-7 rounded shrink-0" style={{ backgroundColor: track.color }} />
									<p className="text-sm font-medium text-white truncate flex-1">{track.title}</p>
									<span className="text-xs text-white/30 font-mono shrink-0">{track.duration}</span>
								</div>
							);
						})
				)}
			</div>
		</div>
	);
}

interface Props {
	tracks: Track[];
	onTransitionComplete: (item: QueueItem) => void;
	onUpdateQueue: (id: string, updates: Partial<QueueItem>) => void;
}

export default function TransitionBuilder({ tracks, onTransitionComplete, onUpdateQueue }: Props) {
	const [fromTrack, setFromTrack] = useState<Track | null>(null);
	const [toTrack, setToTrack] = useState<Track | null>(null);
	const [dragOverFrom, setDragOverFrom] = useState(false);
	const [dragOverTo, setDragOverTo] = useState(false);
	const [isLoading, setIsLoading] = useState(false);
	const [error, setError] = useState<string | null>(null);

	const canGenerate = !!(fromTrack && toTrack && !isLoading);

	function handleDragStart(e: DragEvent, track: Track) {
		e.dataTransfer.setData("text/plain", track.id);
		e.dataTransfer.effectAllowed = "copy";
	}

	function handleDrop(e: DragEvent, target: "from" | "to") {
		e.preventDefault();
		const trackId = e.dataTransfer.getData("text/plain");
		const track = tracks.find((t) => t.id === trackId);
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

	async function handleGenerate() {
		if (!fromTrack || !toTrack) return;
		setIsLoading(true);
		setError(null);

		const itemId = crypto.randomUUID();
		const pendingItem: QueueItem = {
			id: itemId,
			fromTrack,
			toTrack,
			status: "processing",
			transitionDuration: 20,
			createdAt: Date.now(),
		};
		onTransitionComplete(pendingItem);

		try {
			const res = await fetch(`${API_URL}/api/generate`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({
					track1_filename: fromTrack.filename,
					track2_filename: toTrack.filename,
				}),
			});
			if (!res.ok) {
				const err = await res.json();
				throw new Error(err.detail || "Generation failed");
			}
			const data = await res.json();
			const resultUrl = `${API_URL}/api/download/${data.output_filename}`;
			onUpdateQueue(itemId, {
				status: "completed",
				resultUrl,
				outputFilename: data.output_filename,
				transitionDuration: data.transition_duration ?? 20,
				transitionStart: data.transition_start ?? undefined,
			});
		} catch (e: unknown) {
			onUpdateQueue(itemId, { status: "error" });
			setError(e instanceof Error ? e.message : "Generation failed");
		} finally {
			setIsLoading(false);
		}
	}

	return (
		<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="space-y-6">
			<div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
				<TrackPanel
					label="From Track"
					selected={fromTrack}
					tracks={tracks}
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
					tracks={tracks}
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

			{(() => {
				if (!fromTrack || !toTrack) return null;
				const compat = getCompatibility(fromTrack, toTrack);
				if (!compat) return null;
				const bpmDiff = fromTrack.bpm && toTrack.bpm ? Math.abs(fromTrack.bpm - toTrack.bpm) : null;
				return (
					<div className="bento-card px-5 py-4 flex items-center justify-between gap-4">
						<div className="flex items-center gap-3">
							<div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: compat.color }} />
							<div>
								<p className="text-sm font-semibold text-white">{compat.label}</p>
								<p className="text-xs text-white/35">
									{compat.cam1} → {compat.cam2} · {Math.round(compat.score * 100)}% compatible
								</p>
							</div>
						</div>
						{bpmDiff !== null && (
							<div className="text-right shrink-0">
								<p className="text-xs text-white/35 font-mono">
									{fromTrack.bpm} → {toTrack.bpm} BPM
								</p>
								<p className="text-xs text-white/25">{bpmDiff === 0 ? "Matched" : `${bpmDiff} BPM diff`}</p>
							</div>
						)}
					</div>
				);
			})()}

			<button
				type="button"
				disabled={!canGenerate}
				onClick={handleGenerate}
				className={`w-full flex items-center justify-center gap-2.5 py-4 rounded-xl font-semibold text-base transition-all ${canGenerate ? "bg-green-500 text-black hover:bg-green-400 hover:scale-[1.01] shadow-[0_0_40px_-10px_rgba(29,185,84,0.5)]" : "bg-white/5 text-white/20 cursor-not-allowed"}`}
			>
				<SparklesIcon />
				{isLoading ? "Generating… (check Queue tab)" : "Generate Transition"}
			</button>

			{error && <p className="text-sm text-red-400 text-center">{error}</p>}
		</motion.div>
	);
}
