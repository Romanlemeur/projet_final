"use client";

import { useRef, useState, type DragEvent } from "react";
import { motion } from "framer-motion";
import { type Track, TRACK_COLORS, formatDuration } from "@/lib/dashboard-data";

const API_URL = "http://localhost:8000";

function UploadIcon({ spinning }: { spinning?: boolean }) {
	return spinning ? (
		<svg className="w-8 h-8 text-green-500 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
			<circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2" strokeDasharray="31.4 15.7" strokeLinecap="round" />
		</svg>
	) : (
		<svg className="w-8 h-8 text-white/30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
			<polyline points="17 8 12 3 7 8" />
			<line x1="12" y1="3" x2="12" y2="15" />
		</svg>
	);
}

function TrashIcon() {
	return (
		<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<polyline points="3 6 5 6 21 6" />
			<path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
			<path d="M10 11v6M14 11v6" />
			<path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
		</svg>
	);
}

interface Props {
	tracks: Track[];
	onAdd: (track: Track) => void;
	onUpdate: (id: string, updates: Partial<Track>) => void;
	onRemove: (id: string) => void;
}

export default function TrackLibrary({ tracks, onAdd, onUpdate, onRemove }: Props) {
	const inputRef = useRef<HTMLInputElement>(null);
	const [uploading, setUploading] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const [dragOver, setDragOver] = useState(false);
	const [analyzingIds, setAnalyzingIds] = useState<Set<string>>(new Set());

	async function uploadFile(file: File) {
		setUploading(true);
		setError(null);
		try {
			const formData = new FormData();
			formData.append("file", file);
			const res = await fetch(`${API_URL}/api/upload`, { method: "POST", body: formData });
			if (!res.ok) {
				const err = await res.json();
				throw new Error(err.detail || "Upload failed");
			}
			const data = await res.json();

			const objectUrl = URL.createObjectURL(file);
			const durationSeconds = await new Promise<number>((resolve) => {
				const audio = new Audio(objectUrl);
				audio.addEventListener("loadedmetadata", () => {
					resolve(audio.duration);
					URL.revokeObjectURL(objectUrl);
				});
				audio.addEventListener("error", () => {
					resolve(0);
					URL.revokeObjectURL(objectUrl);
				});
			});

			const colorIndex = tracks.length % TRACK_COLORS.length;
			const nameWithoutExt = file.name.replace(/\.[^/.]+$/, "");
			const trackId = crypto.randomUUID();
			const track: Track = {
				id: trackId,
				title: nameWithoutExt,
				artist: "Unknown",
				duration: formatDuration(durationSeconds),
				durationSeconds,
				color: TRACK_COLORS[colorIndex],
				filename: data.filename,
				originalName: file.name,
			};
			onAdd(track);

			setAnalyzingIds((prev) => new Set(prev).add(trackId));
			fetch(`${API_URL}/api/analyze`, {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ filename: data.filename }),
			})
				.then((r) => r.json())
				.then((d) => {
					if (d.analysis) {
						onUpdate(trackId, {
							bpm: Math.round(d.analysis.bpm),
							key: d.analysis.key,
							mode: d.analysis.mode,
						});
					}
				})
				.catch(() => {})
				.finally(() => {
					setAnalyzingIds((prev) => {
						const next = new Set(prev);
						next.delete(trackId);
						return next;
					});
				});
		} catch (e: unknown) {
			setError(e instanceof Error ? e.message : "Upload failed");
		} finally {
			setUploading(false);
		}
	}

	async function handleFiles(files: FileList | null) {
		if (!files || files.length === 0) return;
		for (const file of Array.from(files)) await uploadFile(file);
	}

	function handleDragOver(e: DragEvent) {
		e.preventDefault();
		setDragOver(true);
	}
	function handleDrop(e: DragEvent) {
		e.preventDefault();
		setDragOver(false);
		handleFiles(e.dataTransfer.files);
	}

	return (
		<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="space-y-6">
			<input ref={inputRef} type="file" accept=".mp3,.wav,.flac,.ogg" multiple className="hidden" onChange={(e) => handleFiles(e.target.files)} />

			<button
				type="button"
				disabled={uploading}
				onClick={() => inputRef.current?.click()}
				onDragOver={handleDragOver}
				onDragLeave={() => setDragOver(false)}
				onDrop={handleDrop}
				className={`w-full border-2 border-dashed rounded-2xl p-10 sm:p-14 flex flex-col items-center gap-4 transition-colors cursor-pointer ${dragOver ? "border-green-500/60 bg-green-500/[0.04]" : "border-white/10 hover:border-green-500/40 hover:bg-white/[0.02]"}`}
			>
				<UploadIcon spinning={uploading} />
				<p className="text-white/50 text-base font-medium">
					{uploading ? (
						"Uploading…"
					) : (
						<>
							Drop tracks here or <span className="text-green-500">browse files</span>
						</>
					)}
				</p>
				<p className="text-white/25 text-sm">MP3, WAV, FLAC - up to 50 MB</p>
			</button>

			{error && <p className="text-sm text-red-400 text-center">{error}</p>}

			{tracks.length > 0 && (
				<div className="space-y-2">
					{tracks.map((track) => (
						<div key={track.id} className="flex items-center gap-4 rounded-xl px-4 py-3.5 bg-white/[0.03] border border-white/[0.06] hover:border-white/10 transition-colors group">
							<div className="w-12 h-12 rounded-lg shrink-0 flex items-center justify-center text-black font-bold text-lg" style={{ backgroundColor: track.color }}>
								{track.title.charAt(0).toUpperCase()}
							</div>
							<div className="flex-1 min-w-0">
								<p className="text-sm font-semibold text-white truncate">{track.title}</p>
								<div className="flex items-center gap-2 mt-0.5">
									{analyzingIds.has(track.id) ? (
										<span className="text-xs text-white/25">Analyzing…</span>
									) : track.bpm ? (
										<>
											<span className="text-xs text-white/35 font-mono">{track.bpm} BPM</span>
											<span className="text-white/15">·</span>
											<span className="text-xs text-white/35">
												{track.key} {track.mode}
											</span>
										</>
									) : (
										<span className="text-xs text-white/25 truncate">{track.originalName}</span>
									)}
								</div>
							</div>
							<span className="text-xs text-white/30 font-mono shrink-0">{track.duration}</span>
							<button type="button" onClick={() => onRemove(track.id)} className="text-white/20 hover:text-red-400 transition-colors p-1.5 rounded opacity-0 group-hover:opacity-100" aria-label={`Remove ${track.title}`}>
								<TrashIcon />
							</button>
						</div>
					))}
				</div>
			)}
		</motion.div>
	);
}
