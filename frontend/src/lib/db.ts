import { createClient } from "@/lib/supabase/client";
import { type Track, type QueueItem, type Playlist, formatDuration } from "@/lib/dashboard-data";

const API_URL = "http://localhost:8000";

function mapTrack(row: any): Track {
	return {
		id: row.id,
		title: row.title,
		artist: row.artist,
		filename: row.filename,
		originalName: row.original_name,
		durationSeconds: row.duration_seconds,
		duration: formatDuration(row.duration_seconds),
		color: row.color,
		bpm: row.bpm ?? undefined,
		key: row.key ?? undefined,
		mode: row.mode ?? undefined,
	};
}

function mapQueueItem(row: any): QueueItem {
	return {
		id: row.id,
		fromTrack: mapTrack(row.from_track),
		toTrack: mapTrack(row.to_track),
		status: row.status,
		transitionDuration: row.transition_duration,
		transitionStart: row.transition_start ?? undefined,
		outputFilename: row.output_filename ?? undefined,
		resultUrl: row.output_filename ? `${API_URL}/api/download/${row.output_filename}` : undefined,
		createdAt: new Date(row.created_at).getTime(),
	};
}

function mapPlaylist(row: any): Playlist {
	const items = (row.playlist_items ?? []).sort((a: any, b: any) => a.position - b.position).map((pi: any) => mapQueueItem(pi.queue_items));
	return {
		id: row.id,
		name: row.name,
		color: row.color,
		createdAt: new Date(row.created_at).getTime(),
		items,
	};
}

export async function fetchTracks(): Promise<Track[]> {
	const supabase = createClient();
	const { data } = await supabase.from("tracks").select("*").order("created_at", { ascending: false });
	return (data ?? []).map(mapTrack);
}

export async function insertTrack(track: Track): Promise<void> {
	const supabase = createClient();
	await supabase.from("tracks").insert({
		id: track.id,
		title: track.title,
		artist: track.artist,
		filename: track.filename,
		original_name: track.originalName,
		duration_seconds: track.durationSeconds,
		color: track.color,
		bpm: track.bpm ?? null,
		key: track.key ?? null,
		mode: track.mode ?? null,
	});
}

export async function updateTrack(id: string, updates: Partial<Track>): Promise<void> {
	const supabase = createClient();
	const row: Record<string, unknown> = {};
	if (updates.bpm !== undefined) row.bpm = updates.bpm;
	if (updates.key !== undefined) row.key = updates.key;
	if (updates.mode !== undefined) row.mode = updates.mode;
	if (updates.color !== undefined) row.color = updates.color;
	if (updates.title !== undefined) row.title = updates.title;
	if (updates.artist !== undefined) row.artist = updates.artist;
	if (Object.keys(row).length === 0) return;
	await supabase.from("tracks").update(row).eq("id", id);
}

export async function deleteTrack(id: string): Promise<void> {
	const supabase = createClient();
	await supabase.from("tracks").delete().eq("id", id);
}

export async function fetchQueueItems(): Promise<QueueItem[]> {
	const supabase = createClient();
	const { data } = await supabase.from("queue_items").select("*, from_track:from_track_id(*), to_track:to_track_id(*)").order("created_at", { ascending: false });
	return (data ?? []).map(mapQueueItem);
}

export async function insertQueueItem(item: QueueItem): Promise<void> {
	const supabase = createClient();
	await supabase.from("queue_items").insert({
		id: item.id,
		from_track_id: item.fromTrack.id,
		to_track_id: item.toTrack.id,
		status: item.status,
		transition_duration: item.transitionDuration,
		transition_start: item.transitionStart ?? null,
		output_filename: item.outputFilename ?? null,
	});
}

export async function updateQueueItem(id: string, updates: Partial<QueueItem>): Promise<void> {
	const supabase = createClient();
	const row: Record<string, unknown> = {};
	if (updates.status !== undefined) row.status = updates.status;
	if (updates.transitionDuration !== undefined) row.transition_duration = updates.transitionDuration;
	if (updates.transitionStart !== undefined) row.transition_start = updates.transitionStart;
	if (updates.outputFilename !== undefined) row.output_filename = updates.outputFilename;
	if (Object.keys(row).length === 0) return;
	await supabase.from("queue_items").update(row).eq("id", id);
}

export async function deleteQueueItem(id: string): Promise<void> {
	const supabase = createClient();
	await supabase.from("queue_items").delete().eq("id", id);
}

export async function fetchPlaylists(): Promise<Playlist[]> {
	const supabase = createClient();
	const { data } = await supabase
		.from("playlists")
		.select(
			`
			*,
			playlist_items (
				id,
				position,
				queue_items (
					*,
					from_track:from_track_id(*),
					to_track:to_track_id(*)
				)
			)
		`,
		)
		.order("created_at", { ascending: false });
	return (data ?? []).map(mapPlaylist);
}

export async function insertPlaylist(playlist: Playlist): Promise<void> {
	const supabase = createClient();
	await supabase.from("playlists").insert({ id: playlist.id, name: playlist.name, color: playlist.color });
}

export async function updatePlaylist(id: string, updates: { name?: string; color?: string }): Promise<void> {
	const supabase = createClient();
	await supabase.from("playlists").update(updates).eq("id", id);
}

export async function deletePlaylist(id: string): Promise<void> {
	const supabase = createClient();
	await supabase.from("playlists").delete().eq("id", id);
}

export async function addItemToPlaylist(playlistId: string, queueItemId: string, position: number): Promise<void> {
	const supabase = createClient();
	await supabase.from("playlist_items").insert({ playlist_id: playlistId, queue_item_id: queueItemId, position });
}

export async function removeItemFromPlaylist(playlistId: string, queueItemId: string): Promise<void> {
	const supabase = createClient();
	await supabase.from("playlist_items").delete().eq("playlist_id", playlistId).eq("queue_item_id", queueItemId);
}
