export interface Track {
	id: string;
	title: string;
	artist: string;
	duration: string;
	bpm: number;
	key: string;
	color: string;
}

export interface QueueItem {
	id: string;
	fromTrack: Track;
	toTrack: Track;
	status: "pending" | "processing" | "completed";
	transitionDuration: number;
	fromBpm: number;
	toBpm: number;
}

export const mockTracks: Track[] = [
	{
		id: "t1",
		title: "Midnight Drive",
		artist: "Neon Pulse",
		duration: "3:42",
		bpm: 124,
		key: "Am",
		color: "#1DB954",
	},
	{
		id: "t2",
		title: "Sunset Boulevard",
		artist: "Wave Theory",
		duration: "4:15",
		bpm: 128,
		key: "Cm",
		color: "#6366f1",
	},
	{
		id: "t3",
		title: "Deep Blue",
		artist: "Aqua Drift",
		duration: "5:01",
		bpm: 120,
		key: "Fm",
		color: "#f59e0b",
	},
];

export const mockQueue: QueueItem[] = [
	{
		id: "q1",
		fromTrack: mockTracks[0],
		toTrack: mockTracks[1],
		status: "completed",
		transitionDuration: 8,
		fromBpm: 124,
		toBpm: 128,
	},
];
