"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { motion, AnimatePresence } from "framer-motion";
import { type Playlist, type QueueItem, PLAYLIST_COLORS, formatDuration } from "@/lib/dashboard-data";

function PlusIcon() {
	return (
		<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
			<line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
		</svg>
	);
}
function PencilIcon() {
	return (
		<svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
			<path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
			<path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
		</svg>
	);
}
function TrashIcon({ sm }: { sm?: boolean }) {
	const cls = sm ? "w-3 h-3" : "w-3.5 h-3.5";
	return (
		<svg className={cls} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
			<polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" /><path d="M10 11v6M14 11v6" />
		</svg>
	);
}
function ChevronIcon({ open }: { open: boolean }) {
	return (
		<svg className={`w-4 h-4 transition-transform ${open ? "rotate-180" : ""}`} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
			<polyline points="6 9 12 15 18 9" />
		</svg>
	);
}
function PlayIcon({ size = "sm" }: { size?: "sm" | "md" }) {
	const cls = size === "md" ? "w-5 h-5 translate-x-px" : "w-3 h-3 translate-x-px";
	return <svg className={cls} viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21 5 3" /></svg>;
}
function PauseIcon({ size = "sm" }: { size?: "sm" | "md" }) {
	const cls = size === "md" ? "w-5 h-5" : "w-3 h-3";
	return <svg className={cls} viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16" rx="1" /><rect x="14" y="4" width="4" height="16" rx="1" /></svg>;
}
function SkipBackIcon() {
	return <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><polygon points="19 20 9 12 19 4 19 20" /><line x1="5" y1="19" x2="5" y2="5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>;
}
function SkipForwardIcon() {
	return <svg className="w-4 h-4" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 4 15 12 5 20 5 4" /><line x1="19" y1="5" x2="19" y2="19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" /></svg>;
}
function ShuffleIcon() {
	return (
		<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
			<polyline points="16 3 21 3 21 8" />
			<line x1="4" y1="20" x2="21" y2="3" />
			<polyline points="21 16 21 21 16 21" />
			<line x1="15" y1="15" x2="21" y2="21" />
		</svg>
	);
}

function ColorPicker({ value, onChange }: { value: string; onChange: (c: string) => void }) {
	return (
		<div className="flex gap-3 flex-wrap p-1">
			{PLAYLIST_COLORS.map((c) => (
				<button key={c} type="button" onClick={() => onChange(c)}
					className={`w-6 h-6 rounded-full transition-transform hover:scale-110 ${value === c ? "ring-2 ring-white ring-offset-2 ring-offset-[#111]" : ""}`}
					style={{ backgroundColor: c }} />
			))}
		</div>
	);
}

function CreatePlaylistForm({ onCreate }: { onCreate: (name: string) => void }) {
	const [name, setName] = useState("");
	const [open, setOpen] = useState(false);
	function submit() {
		const t = name.trim(); if (!t) return;
		onCreate(t); setName(""); setOpen(false);
	}
	if (!open) return (
		<button type="button" onClick={() => setOpen(true)}
			className="flex items-center gap-2 px-4 py-3 rounded-xl border-2 border-dashed border-white/10 text-white/40 hover:border-green-500/40 hover:text-green-500 transition-colors w-full text-sm font-medium">
			<PlusIcon />New Playlist
		</button>
	);
	return (
		<motion.div initial={{ opacity: 0, y: -4 }} animate={{ opacity: 1, y: 0 }} className="bento-card p-5 space-y-4">
			<p className="text-sm font-medium text-white">Create Playlist</p>
			<input autoFocus value={name} onChange={(e) => setName(e.target.value)}
				onKeyDown={(e) => { if (e.key === "Enter") submit(); if (e.key === "Escape") setOpen(false); }}
				placeholder="Playlist name…"
				className="w-full bg-white/5 rounded-lg px-4 py-2.5 text-sm text-white placeholder-white/25 outline-none border border-white/10 focus:border-green-500/50 transition-colors" />
			<div className="flex gap-2 justify-end">
				<button type="button" onClick={() => setOpen(false)} className="px-4 py-2 rounded-lg text-sm text-white/40 hover:text-white transition-colors">Cancel</button>
				<button type="button" onClick={submit} disabled={!name.trim()} className="px-4 py-2 rounded-lg text-sm font-medium bg-green-500 text-black hover:bg-green-400 disabled:opacity-40 disabled:cursor-not-allowed transition-colors">Create</button>
			</div>
		</motion.div>
	);
}

function EditPlaylistForm({ playlist, onRename, onChangeColor, onClose }: {
	playlist: Playlist; onRename: (id: string, name: string) => void;
	onChangeColor: (id: string, color: string) => void; onClose: () => void;
}) {
	const [name, setName] = useState(playlist.name);
	const [color, setColor] = useState(playlist.color);
	function save() {
		const t = name.trim(); if (t) onRename(playlist.id, t);
		onChangeColor(playlist.id, color); onClose();
	}
	return (
		<motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
			<div className="pt-3 pb-1 space-y-4 border-t border-white/[0.06] mt-3">
				<input autoFocus value={name} onChange={(e) => setName(e.target.value)}
					onKeyDown={(e) => { if (e.key === "Enter") save(); if (e.key === "Escape") onClose(); }}
					className="w-full bg-white/5 rounded-lg px-3 py-2 text-sm text-white outline-none border border-white/10 focus:border-green-500/50 transition-colors" />
				<ColorPicker value={color} onChange={setColor} />
				<div className="flex gap-2 justify-end">
					<button type="button" onClick={onClose} className="px-3 py-1.5 rounded-lg text-xs text-white/40 hover:text-white transition-colors">Cancel</button>
					<button type="button" onClick={save} className="px-3 py-1.5 rounded-lg text-xs font-medium bg-white/10 hover:bg-white/20 text-white transition-colors">Save</button>
				</div>
			</div>
		</motion.div>
	);
}

function PlaylistItemRow({ item, isActive, isPlaying, onPlay, onRemove }: {
	item: QueueItem; isActive: boolean; isPlaying: boolean; onPlay: () => void; onRemove: () => void;
}) {
	return (
		<div className={`flex items-center gap-3 py-2.5 px-3 rounded-lg group transition-colors ${isActive ? "bg-white/[0.05]" : "hover:bg-white/[0.03]"}`}>
			<div className="w-7 h-7 rounded shrink-0 flex items-center justify-center text-[10px] text-black font-bold" style={{ backgroundColor: item.fromTrack.color }}>
				{isActive && isPlaying
					? <span className="flex gap-px items-end h-3">
							<span className="w-px bg-black animate-bounce" style={{ height: "40%", animationDelay: "0ms" }} />
							<span className="w-px bg-black animate-bounce" style={{ height: "70%", animationDelay: "150ms" }} />
							<span className="w-px bg-black animate-bounce" style={{ height: "55%", animationDelay: "75ms" }} />
						</span>
					: item.fromTrack.title.charAt(0)}
			</div>
			<span className={`text-xs truncate flex-1 ${isActive ? "text-green-400 font-medium" : "text-white/60"}`}>
				{item.fromTrack.title} <span className="text-white/25 mx-1">→</span> {item.toTrack.title}
			</span>
			{item.resultUrl && (
				<button type="button" onClick={onPlay} title="Play"
					className={`p-1 transition-colors ${isActive ? "text-green-400" : "text-white/20 hover:text-green-400 opacity-0 group-hover:opacity-100"}`}>
					<PlayIcon />
				</button>
			)}
			<button type="button" onClick={onRemove} aria-label="Remove from playlist"
				className="text-white/15 hover:text-red-400 transition-colors p-1 opacity-0 group-hover:opacity-100">
				<TrashIcon />
			</button>
		</div>
	);
}

function PlaylistCard({ playlist, activeItemId, isPlaying, onPlayItem, onRename, onChangeColor, onDelete, onRemoveItem }: {
	playlist: Playlist; activeItemId: string | null; isPlaying: boolean;
	onPlayItem: (playlist: Playlist, itemIdx: number) => void;
	onRename: (id: string, name: string) => void; onChangeColor: (id: string, color: string) => void;
	onDelete: (id: string) => void; onRemoveItem: (playlistId: string, itemId: string) => void;
}) {
	const [expanded, setExpanded] = useState(false);
	const [editing, setEditing] = useState(false);
	const [confirmDelete, setConfirmDelete] = useState(false);
	const totalDuration = playlist.items.reduce((acc, i) => acc + i.transitionDuration, 0);

	return (
		<div className="bento-card overflow-hidden">
			<div className="flex items-center gap-4 p-5">
				<div className="w-12 h-12 rounded-xl shrink-0 flex items-center justify-center font-bold text-black text-lg" style={{ backgroundColor: playlist.color }}>
					{playlist.name.charAt(0).toUpperCase()}
				</div>
				<div className="flex-1 min-w-0">
					<p className="text-sm font-semibold text-white truncate">{playlist.name}</p>
					<p className="text-xs text-white/35 mt-0.5">
						{playlist.items.length} transition{playlist.items.length !== 1 ? "s" : ""}
						{totalDuration > 0 && <> · {formatDuration(totalDuration)}</>}
					</p>
				</div>
				<div className="flex items-center gap-1 shrink-0">
					<button type="button" onClick={() => { setEditing((e) => !e); setExpanded(true); }} aria-label="Edit playlist"
						className="p-2 rounded-lg text-white/30 hover:text-white hover:bg-white/[0.06] transition-colors">
						<PencilIcon />
					</button>
					{confirmDelete ? (
						<div className="flex items-center gap-2 ml-1">
							<span className="text-xs text-white/40">Delete?</span>
							<button type="button" onClick={() => onDelete(playlist.id)} className="text-xs text-red-400 font-medium hover:text-red-300">Yes</button>
							<button type="button" onClick={() => setConfirmDelete(false)} className="text-xs text-white/40 hover:text-white">No</button>
						</div>
					) : (
						<button type="button" onClick={() => setConfirmDelete(true)} aria-label="Delete playlist"
							className="p-2 rounded-lg text-white/30 hover:text-red-400 hover:bg-white/[0.06] transition-colors">
							<TrashIcon />
						</button>
					)}
					<button type="button" onClick={() => setExpanded((e) => !e)} aria-label="Expand"
						className="p-2 rounded-lg text-white/30 hover:text-white hover:bg-white/[0.06] transition-colors">
						<ChevronIcon open={expanded} />
					</button>
				</div>
			</div>

			<AnimatePresence>
				{editing && (
					<div className="px-5 pb-2">
						<EditPlaylistForm playlist={playlist} onRename={onRename} onChangeColor={onChangeColor} onClose={() => setEditing(false)} />
					</div>
				)}
			</AnimatePresence>

			<AnimatePresence>
				{expanded && !editing && (
					<motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }} className="overflow-hidden">
						<div className="border-t border-white/[0.06] px-3 py-2">
							{playlist.items.length === 0 ? (
								<p className="text-xs text-white/20 text-center py-4">No transitions yet. Add some from the Queue tab.</p>
							) : (
								playlist.items.map((item, idx) => (
									<PlaylistItemRow
										key={item.id}
										item={item}
										isActive={activeItemId === item.id}
										isPlaying={isPlaying && activeItemId === item.id}
										onPlay={() => onPlayItem(playlist, idx)}
										onRemove={() => onRemoveItem(playlist.id, item.id)}
									/>
								))
							)}
						</div>
					</motion.div>
				)}
			</AnimatePresence>
		</div>
	);
}

function BottomPlayerBar({ activeItem, playlistName, playlistColor, playing, shuffle, currentTime, duration, onTogglePlay, onPrev, onNext, onToggleShuffle, onSeek }: {
	activeItem: QueueItem | null; playlistName: string; playlistColor: string;
	playing: boolean; shuffle: boolean; currentTime: number; duration: number;
	onTogglePlay: () => void; onPrev: () => void; onNext: () => void;
	onToggleShuffle: () => void; onSeek: (t: number) => void;
}) {
	function handleSeekClick(e: React.MouseEvent<HTMLDivElement>) {
		if (!duration) return;
		const rect = e.currentTarget.getBoundingClientRect();
		onSeek(((e.clientX - rect.left) / rect.width) * duration);
	}

	return createPortal(
		<AnimatePresence>
			{activeItem && (
				<motion.div
					initial={{ y: 80, opacity: 0 }}
					animate={{ y: 0, opacity: 1 }}
					exit={{ y: 80, opacity: 0 }}
					transition={{ type: "spring", stiffness: 300, damping: 30 }}
					className="fixed bottom-0 left-0 right-0 z-50 border-t border-white/[0.08] bg-[#0a0a0a]/95 backdrop-blur-xl px-6 py-3"
				>
					<div className="w-full mb-3 flex items-center gap-3">
						<span className="text-[10px] text-white/30 font-mono tabular-nums w-8 text-right shrink-0">{formatDuration(currentTime)}</span>
						<div className="flex-1 h-1 bg-white/10 rounded-full cursor-pointer relative group" onClick={handleSeekClick}>
							<div className="absolute left-0 top-0 h-full bg-green-500 rounded-full transition-none" style={{ width: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%" }} />
							<div className="absolute top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full opacity-0 group-hover:opacity-100 transition-opacity -ml-1.5 pointer-events-none"
								style={{ left: duration > 0 ? `${(currentTime / duration) * 100}%` : "0%" }} />
						</div>
						<span className="text-[10px] text-white/30 font-mono tabular-nums w-8 shrink-0">{formatDuration(duration)}</span>
					</div>

					<div className="flex items-center gap-4">
						<div className="flex items-center gap-3 min-w-0 flex-1">
							<div className="w-9 h-9 rounded-lg shrink-0 flex items-center justify-center text-xs font-bold text-black" style={{ backgroundColor: playlistColor }}>
								{playlistName.charAt(0).toUpperCase()}
							</div>
							<div className="min-w-0">
								<p className="text-xs font-medium text-white truncate">
									{activeItem.fromTrack.title} <span className="text-white/30">→</span> {activeItem.toTrack.title}
								</p>
								<p className="text-[10px] text-white/30 truncate">{playlistName}</p>
							</div>
						</div>

						<div className="flex items-center gap-3 shrink-0">
							<button type="button" onClick={onToggleShuffle} aria-label="Shuffle"
								className={`p-1.5 rounded-lg transition-colors ${shuffle ? "text-green-400" : "text-white/25 hover:text-white/60"}`}>
								<ShuffleIcon />
							</button>
							<button type="button" onClick={onPrev} aria-label="Previous"
								className="p-1.5 text-white/50 hover:text-white transition-colors">
								<SkipBackIcon />
							</button>
							<button type="button" onClick={onTogglePlay} aria-label={playing ? "Pause" : "Play"}
								className="w-10 h-10 rounded-full bg-white text-black flex items-center justify-center hover:bg-white/80 transition-colors">
								{playing ? <PauseIcon size="md" /> : <PlayIcon size="md" />}
							</button>
							<button type="button" onClick={onNext} aria-label="Next"
								className="p-1.5 text-white/50 hover:text-white transition-colors">
								<SkipForwardIcon />
							</button>
							<div className="w-7" />
						</div>

						<div className="flex-1" />
					</div>
				</motion.div>
			)}
		</AnimatePresence>,
		document.body
	);
}

interface Props {
	playlists: Playlist[];
	onCreate: (name: string) => void;
	onRename: (id: string, name: string) => void;
	onChangeColor: (id: string, color: string) => void;
	onDelete: (id: string) => void;
	onRemoveItem: (playlistId: string, itemId: string) => void;
}

export default function PlaylistManager({ playlists, onCreate, onRename, onChangeColor, onDelete, onRemoveItem }: Props) {
	const audioRef = useRef<HTMLAudioElement>(null);
	const rafRef = useRef<number>(0);
	const [activePl, setActivePl] = useState<Playlist | null>(null);
	const [activeIdx, setActiveIdx] = useState<number | null>(null);
	const [playing, setPlaying] = useState(false);
	const [shuffle, setShuffle] = useState(false);
	const [shuffleOrder, setShuffleOrder] = useState<number[]>([]);
	const [currentTime, setCurrentTime] = useState(0);
	const [duration, setDuration] = useState(0);

	const activeItem = activePl && activeIdx !== null ? (activePl.items[activeIdx] ?? null) : null;

	function buildShuffleOrder(items: QueueItem[], startIdx: number): number[] {
		const rest = items.map((_, i) => i).filter((i) => i !== startIdx);
		for (let i = rest.length - 1; i > 0; i--) {
			const j = Math.floor(Math.random() * (i + 1));
			[rest[i], rest[j]] = [rest[j], rest[i]];
		}
		return [startIdx, ...rest];
	}

	function playItem(pl: Playlist, idx: number) {
		const newOrder = buildShuffleOrder(pl.items, idx);
		if (shuffle) setShuffleOrder(newOrder);
		setActivePl(pl);
		setActiveIdx(idx);
		setPlaying(true);
	}

	function navigate(dir: 1 | -1) {
		if (!activePl || activeIdx === null) return;
		const items = activePl.items;
		if (items.length === 0) return;
		const order = shuffle ? shuffleOrder : items.map((_, i) => i);
		const pos = order.indexOf(activeIdx);
		const next = order[(pos + dir + order.length) % order.length];
		setActiveIdx(next);
		setPlaying(true);
	}

	function toggleShuffle() {
		if (!shuffle && activePl && activeIdx !== null) {
			setShuffleOrder(buildShuffleOrder(activePl.items, activeIdx));
		}
		setShuffle((s) => !s);
	}

	useEffect(() => {
		const audio = audioRef.current;
		if (!audio) return;
		const url = activeItem?.resultUrl ?? null;
		if (!url) { audio.pause(); return; }
		if (audio.src !== url) {
			audio.src = url;
			setCurrentTime(0);
			setDuration(0);
		}
		if (playing) audio.play().catch(() => {});
		else audio.pause();
	}, [activeItem, playing]);

	useEffect(() => {
		function tick() {
			const a = audioRef.current;
			if (a) setCurrentTime(a.currentTime);
			rafRef.current = requestAnimationFrame(tick);
		}
		if (playing) { rafRef.current = requestAnimationFrame(tick); }
		else { cancelAnimationFrame(rafRef.current); }
		return () => cancelAnimationFrame(rafRef.current);
	}, [playing]);

	return (
		<>
			<audio
				ref={audioRef}
				onLoadedMetadata={(e) => setDuration((e.target as HTMLAudioElement).duration)}
				onEnded={() => navigate(1)}
				preload="none"
			/>

			<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}
				className={`space-y-4 ${activeItem ? "pb-28" : ""}`}>
				<CreatePlaylistForm onCreate={onCreate} />
				{playlists.map((pl) => (
					<PlaylistCard
						key={pl.id}
						playlist={pl}
						activeItemId={activePl?.id === pl.id ? (activeItem?.id ?? null) : null}
						isPlaying={playing}
						onPlayItem={playItem}
						onRename={onRename}
						onChangeColor={onChangeColor}
						onDelete={onDelete}
						onRemoveItem={onRemoveItem}
					/>
				))}
			</motion.div>

			<BottomPlayerBar
				activeItem={activeItem}
				playlistName={activePl?.name ?? ""}
				playlistColor={activePl?.color ?? "#1DB954"}
				playing={playing}
				shuffle={shuffle}
				currentTime={currentTime}
				duration={duration}
				onTogglePlay={() => setPlaying((p) => !p)}
				onPrev={() => navigate(-1)}
				onNext={() => navigate(1)}
				onToggleShuffle={toggleShuffle}
				onSeek={(t) => { if (audioRef.current) { audioRef.current.currentTime = t; setCurrentTime(t); } }}
			/>
		</>
	);
}
