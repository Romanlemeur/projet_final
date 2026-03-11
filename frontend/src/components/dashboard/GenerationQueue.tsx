"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion } from "framer-motion";
import { type QueueItem, type Playlist, formatDuration } from "@/lib/dashboard-data";

function PlayIcon() {
	return (
		<svg className="w-5 h-5 translate-x-px" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
			<polygon points="5 3 19 12 5 21 5 3" />
		</svg>
	);
}
function PauseIcon() {
	return (
		<svg className="w-5 h-5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
			<rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" />
		</svg>
	);
}
function DownloadIcon() {
	return (
		<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="3" x2="12" y2="15" />
		</svg>
	);
}
function TrashIcon() {
	return (
		<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" /><path d="M10 11v6M14 11v6" />
		</svg>
	);
}
function PlusIcon() {
	return (
		<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
		</svg>
	);
}

const peaksCache = new Map<string, number[]>();

function WaveformPlayer({ url, transitionStart, transitionDuration }: { url: string; transitionStart?: number; transitionDuration?: number }) {
	const canvasRef = useRef<HTMLCanvasElement>(null);
	const audioRef = useRef<HTMLAudioElement>(null);
	const rafRef = useRef<number>(0);
	const [peaks, setPeaks] = useState<number[]>(() => peaksCache.get(url) ?? []);
	const [playing, setPlaying] = useState(false);
	const [currentTime, setCurrentTime] = useState(0);
	const [duration, setDuration] = useState(0);

	useEffect(() => {
		if (peaksCache.has(url)) return;
		let cancelled = false;
		async function load() {
			try {
				const res = await fetch(url);
				const buf = await res.arrayBuffer();
				const ctx = new OfflineAudioContext(1, 1, 44100);
				const decoded = await ctx.decodeAudioData(buf);
				if (cancelled) return;
				const data = decoded.getChannelData(0);
				const N = 200;
				const block = Math.floor(data.length / N);
				const p: number[] = [];
				for (let i = 0; i < N; i++) {
					let max = 0;
					for (let j = 0; j < block; j++) max = Math.max(max, Math.abs(data[i * block + j]));
					p.push(max);
				}
				peaksCache.set(url, p);
				setPeaks(p);
			} catch { /* CORS or decode error — fall back to decorative bars */ }
		}
		load();
		return () => { cancelled = true; };
	}, [url]);

	useEffect(() => {
		const canvas = canvasRef.current;
		if (!canvas) return;
		const ctx = canvas.getContext("2d");
		if (!ctx) return;
		const W = canvas.width, H = canvas.height;
		ctx.clearRect(0, 0, W, H);

		if (peaks.length === 0) return;

		const barCount = peaks.length;
		const progress = duration > 0 ? currentTime / duration : 0;
		const src = peaks;
		const barW = W / barCount - 1;

		const tzStart = duration > 0 && transitionStart != null ? transitionStart / duration : null;
		const tzEnd = tzStart != null && transitionDuration != null ? tzStart + transitionDuration / duration : null;

		if (tzStart != null && tzEnd != null) {
			ctx.fillStyle = "rgba(99,102,241,0.10)";
			ctx.fillRect(tzStart * W, 0, (tzEnd - tzStart) * W, H);
		}

		src.forEach((h, i) => {
			const x = i * (barW + 1);
			const barH = Math.max(2, h * H * 0.9);
			const y = (H - barH) / 2;
			const frac = i / barCount;
			const played = frac < progress;
			const inTransition = tzStart != null && tzEnd != null && frac >= tzStart && frac < tzEnd;
			if (inTransition) {
				ctx.fillStyle = played ? "rgba(129,140,248,0.95)" : "rgba(99,102,241,0.45)";
			} else {
				ctx.fillStyle = played ? "#1DB954" : "rgba(255,255,255,0.15)";
			}
			ctx.fillRect(x, y, Math.max(1, barW), barH);
		});
	}, [peaks, currentTime, duration, transitionStart, transitionDuration]);

	useEffect(() => {
		function tick() {
			const a = audioRef.current;
			if (a) setCurrentTime(a.currentTime);
			rafRef.current = requestAnimationFrame(tick);
		}
		if (playing) {
			rafRef.current = requestAnimationFrame(tick);
		} else {
			cancelAnimationFrame(rafRef.current);
		}
		return () => cancelAnimationFrame(rafRef.current);
	}, [playing]);

	function handleCanvasClick(e: React.MouseEvent<HTMLCanvasElement>) {
		const canvas = canvasRef.current;
		const audio = audioRef.current;
		if (!canvas || !audio || !duration) return;
		const rect = canvas.getBoundingClientRect();
		const ratio = (e.clientX - rect.left) / rect.width;
		audio.currentTime = ratio * duration;
		setCurrentTime(audio.currentTime);
	}

	function togglePlay() {
		const audio = audioRef.current;
		if (!audio) return;
		if (playing) { audio.pause(); setPlaying(false); }
		else { audio.play(); setPlaying(true); }
	}

	return (
		<div className="space-y-3">
			<audio
				ref={audioRef}
				src={url}
				onLoadedMetadata={(e) => setDuration((e.target as HTMLAudioElement).duration)}
				onEnded={() => setPlaying(false)}
				preload="metadata"
			/>

			<canvas
				ref={canvasRef}
				width={800}
				height={64}
				onClick={handleCanvasClick}
				className="w-full h-16 cursor-pointer rounded-lg"
			/>

			<div className="flex items-center gap-3">
				<button
					type="button"
					onClick={togglePlay}
					className="w-9 h-9 rounded-full bg-green-500 text-black flex items-center justify-center hover:bg-green-400 transition-colors shrink-0"
				>
					{playing ? <PauseIcon /> : <PlayIcon />}
				</button>

				<span className="text-xs text-white/40 font-mono tabular-nums w-20 shrink-0">
					{formatDuration(currentTime)} / {formatDuration(duration)}
				</span>

				<div
					className="flex-1 h-1 bg-white/10 rounded-full cursor-pointer relative"
					onClick={(e) => {
						const audio = audioRef.current;
						if (!audio || !duration) return;
						const rect = e.currentTarget.getBoundingClientRect();
						audio.currentTime = ((e.clientX - rect.left) / rect.width) * duration;
					}}
				>
					<div
						className="absolute left-0 top-0 h-full bg-green-500 rounded-full"
						style={{ width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%" }}
					/>
				</div>
			</div>
		</div>
	);
}

function AddToPlaylistMenu({
	item, playlists, onAddToPlaylist, onCreateAndAddToPlaylist,
}: {
	item: QueueItem;
	playlists: Playlist[];
	onAddToPlaylist: (playlistId: string, item: QueueItem) => void;
	onCreateAndAddToPlaylist: (name: string, item: QueueItem) => void;
}) {
	const [open, setOpen] = useState(false);
	const [creating, setCreating] = useState(false);
	const [newName, setNewName] = useState("");
	const [pos, setPos] = useState({ top: 0, right: 0 });
	const btnRef = useRef<HTMLButtonElement>(null);
	const menuRef = useRef<HTMLDivElement>(null);

	function toggle() {
		if (open) { setOpen(false); setCreating(false); return; }
		if (btnRef.current) {
			const r = btnRef.current.getBoundingClientRect();
			setPos({ top: r.bottom + 6, right: window.innerWidth - r.right });
		}
		setOpen(true);
	}

	useEffect(() => {
		if (!open) return;
		function onMouseDown(e: MouseEvent) {
			const t = e.target as Node;
			if (!btnRef.current?.contains(t) && !menuRef.current?.contains(t)) {
				setOpen(false);
				setCreating(false);
			}
		}
		document.addEventListener("mousedown", onMouseDown);
		return () => document.removeEventListener("mousedown", onMouseDown);
	}, [open]);

	function handleCreate() {
		const name = newName.trim();
		if (!name) return;
		onCreateAndAddToPlaylist(name, item);
		setNewName("");
		setCreating(false);
		setOpen(false);
	}

	return (
		<>
			<button
				ref={btnRef}
				type="button"
				onClick={toggle}
				className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-xs font-medium transition-colors"
			>
				<PlusIcon />
				Add to Playlist
			</button>

			{open && createPortal(
				<div
					ref={menuRef}
					style={{ position: "fixed", top: pos.top, right: pos.right, zIndex: 9999 }}
					className="w-52 rounded-xl bg-[#1a1a1a] border border-white/10 shadow-2xl overflow-hidden"
				>
					{playlists.length === 0 && !creating && (
						<p className="text-xs text-white/30 px-4 py-3">No playlists yet</p>
					)}
					{playlists.map((pl) => {
						const already = pl.items.some((i) => i.id === item.id);
						return (
							<button
								key={pl.id}
								type="button"
								disabled={already}
								onClick={() => { onAddToPlaylist(pl.id, item); setOpen(false); }}
								className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm text-left transition-colors ${already ? "opacity-40 cursor-not-allowed" : "hover:bg-white/[0.06] text-white"}`}
							>
								<div className="w-3 h-3 rounded-full shrink-0" style={{ backgroundColor: pl.color }} />
								<span className="truncate">{pl.name}</span>
								{already && <span className="ml-auto text-[10px] text-white/30">Added</span>}
							</button>
						);
					})}

					<div className="border-t border-white/[0.06] px-3 py-2">
						{creating ? (
							<div className="flex gap-2">
								<input
									autoFocus
									value={newName}
									onChange={(e) => setNewName(e.target.value)}
									onKeyDown={(e) => { if (e.key === "Enter") handleCreate(); if (e.key === "Escape") setCreating(false); }}
									placeholder="Playlist name…"
									className="flex-1 bg-white/5 rounded px-2 py-1 text-xs text-white outline-none border border-white/10 focus:border-green-500/50"
								/>
								<button type="button" onClick={handleCreate} className="text-green-500 text-xs font-medium hover:text-green-400">Create</button>
							</div>
						) : (
							<button
								type="button"
								onClick={() => setCreating(true)}
								className="w-full flex items-center gap-2 text-xs text-white/40 hover:text-white transition-colors py-1"
							>
								<PlusIcon />
								New Playlist
							</button>
						)}
					</div>
				</div>,
				document.body
			)}
		</>
	);
}

function StatusBadge({ status }: { status: QueueItem["status"] }) {
	if (status === "completed") return (
		<span className="flex items-center gap-1.5 text-green-500 text-xs font-medium">
			<svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12" /></svg>
			Done
		</span>
	);
	if (status === "processing") return (
		<span className="flex items-center gap-1.5 text-green-500 text-xs font-medium">
			<svg className="w-3.5 h-3.5 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="2.5" strokeDasharray="31.4 15.7" strokeLinecap="round" /></svg>
			Generating…
		</span>
	);
	if (status === "error") return (
		<span className="flex items-center gap-1.5 text-red-400 text-xs font-medium">
			<svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" /></svg>
			Failed
		</span>
	);
	return (
		<span className="flex items-center gap-1.5 text-white/30 text-xs font-medium">
			<svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10" /><polyline points="12 6 12 12 16 14" /></svg>
			Queued
		</span>
	);
}

interface Props {
	items: QueueItem[];
	playlists: Playlist[];
	onAddToPlaylist: (playlistId: string, item: QueueItem) => void;
	onRemove: (id: string) => void;
	onCreatePlaylist: (name: string) => void;
	onCreateAndAddToPlaylist: (name: string, item: QueueItem) => void;
}

export default function GenerationQueue({ items, playlists, onAddToPlaylist, onRemove, onCreatePlaylist, onCreateAndAddToPlaylist }: Props) {
	if (items.length === 0) {
		return (
			<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="flex flex-col items-center justify-center py-20 text-center">
				<p className="text-white/30 text-sm">No transitions yet.</p>
				<p className="text-white/20 text-xs mt-1">Generate one in the Builder tab.</p>
			</motion.div>
		);
	}

	return (
		<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="space-y-4">
			{items.map((item) => (
				<div key={item.id} className="bento-card p-5 sm:p-6 space-y-4">
					<div className="flex items-center justify-between gap-4">
						<div className="flex items-center gap-2 min-w-0">
							<div className="w-8 h-8 rounded-md shrink-0 flex items-center justify-center text-black text-xs font-bold" style={{ backgroundColor: item.fromTrack.color }}>
								{item.fromTrack.title.charAt(0)}
							</div>
							<span className="text-sm font-medium text-white truncate max-w-[120px]">{item.fromTrack.title}</span>
							<span className="text-white/20 text-xs shrink-0">→</span>
							<div className="w-8 h-8 rounded-md shrink-0 flex items-center justify-center text-black text-xs font-bold" style={{ backgroundColor: item.toTrack.color }}>
								{item.toTrack.title.charAt(0)}
							</div>
							<span className="text-sm font-medium text-white truncate max-w-[120px]">{item.toTrack.title}</span>
						</div>
						<div className="flex items-center gap-3 shrink-0">
							<StatusBadge status={item.status} />
							<button
								type="button"
								onClick={() => onRemove(item.id)}
								className="text-white/20 hover:text-red-400 transition-colors p-1 rounded"
								aria-label="Remove"
							>
								<TrashIcon />
							</button>
						</div>
					</div>

					{item.status === "completed" && item.resultUrl ? (
						<WaveformPlayer url={item.resultUrl} transitionStart={item.transitionStart} transitionDuration={item.transitionDuration} />
					) : null}

					{item.status === "completed" && item.resultUrl && (
						<div className="flex items-center justify-end gap-3 pt-1">
							<div className="flex items-center gap-2">
								<AddToPlaylistMenu
									item={item}
									playlists={playlists}
									onAddToPlaylist={onAddToPlaylist}
									onCreateAndAddToPlaylist={onCreateAndAddToPlaylist}
								/>
								<a
									href={item.resultUrl}
									download={`transition-${item.fromTrack.title}-${item.toTrack.title}.wav`}
									className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white/5 hover:bg-white/10 text-white/60 hover:text-white text-xs font-medium transition-colors"
								>
									<DownloadIcon />
									Download
								</a>
							</div>
						</div>
					)}
				</div>
			))}
		</motion.div>
	);
}
