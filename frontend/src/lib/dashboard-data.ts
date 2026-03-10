export interface Track {
	id: string;
	title: string;
	artist: string;
	duration: string;
	durationSeconds: number;
	color: string;
	filename: string;
	originalName: string;
	bpm?: number;
	key?: string;
	mode?: string;
}

export interface QueueItem {
	id: string;
	fromTrack: Track;
	toTrack: Track;
	status: "pending" | "processing" | "completed" | "error";
	transitionDuration: number;
	transitionStart?: number;
	resultUrl?: string;
	outputFilename?: string;
	createdAt: number;
}

export interface Playlist {
	id: string;
	name: string;
	color: string;
	items: QueueItem[];
	createdAt: number;
}

export const TRACK_COLORS = [
	"#1DB954", "#6366f1", "#f59e0b", "#ef4444",
	"#06b6d4", "#8b5cf6", "#ec4899", "#10b981",
];

export const PLAYLIST_COLORS = [
	"#1DB954", "#6366f1", "#f59e0b", "#ef4444",
	"#06b6d4", "#8b5cf6", "#ec4899", "#f97316",
];

export function formatDuration(totalSeconds: number): string {
	const m = Math.floor(totalSeconds / 60);
	const s = Math.floor(totalSeconds % 60);
	return `${m}:${s.toString().padStart(2, "0")}`;
}

export function stringToColor(str: string): string {
	let hash = 0;
	for (let i = 0; i < str.length; i++) {
		hash = str.charCodeAt(i) + ((hash << 5) - hash);
	}
	return TRACK_COLORS[Math.abs(hash) % TRACK_COLORS.length];
}
