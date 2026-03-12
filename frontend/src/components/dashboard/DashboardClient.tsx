"use client";

import { useState, useEffect, useRef } from "react";
import { AnimatePresence } from "framer-motion";
import DashboardHeader from "./DashboardHeader";
import DashboardTabs, { type TabId } from "./DashboardTabs";
import TrackLibrary from "./TrackLibrary";
import TransitionBuilder from "./TransitionBuilder";
import GenerationQueue from "./GenerationQueue";
import PlaylistManager from "./PlaylistManager";
import { type Track, type QueueItem, type Playlist, PLAYLIST_COLORS } from "@/lib/dashboard-data";
import * as db from "@/lib/db";

export default function DashboardClient() {
	const [activeTab, setActiveTab] = useState<TabId>("library");
	const [tracks, setTracks] = useState<Track[]>([]);
	const [queueItems, setQueueItems] = useState<QueueItem[]>([]);
	const [playlists, setPlaylists] = useState<Playlist[]>([]);
	const [loading, setLoading] = useState(true);

	const pendingInserts = useRef<Map<string, Promise<void>>>(new Map());

	useEffect(() => {
		Promise.all([db.fetchTracks(), db.fetchQueueItems(), db.fetchPlaylists()]).then(([t, q, p]) => {
			setTracks(t);
			setQueueItems(q);
			setPlaylists(p);
			setLoading(false);
		});
	}, []);

	function addTrack(track: Track) {
		setTracks((prev) => [track, ...prev]);
		db.insertTrack(track);
	}

	function updateTrack(id: string, updates: Partial<Track>) {
		setTracks((prev) => prev.map((t) => (t.id === id ? { ...t, ...updates } : t)));
		db.updateTrack(id, updates);
	}

	function removeTrack(id: string) {
		setTracks((prev) => prev.filter((t) => t.id !== id));
		db.deleteTrack(id);
	}

	function addQueueItem(item: QueueItem) {
		setQueueItems((prev) => [item, ...prev]);
		setActiveTab("queue");
		const insert = db.insertQueueItem(item);
		pendingInserts.current.set(item.id, insert);
		insert.finally(() => pendingInserts.current.delete(item.id));
	}

	function updateQueueItem(id: string, updates: Partial<QueueItem>) {
		setQueueItems((prev) => prev.map((q) => (q.id === id ? { ...q, ...updates } : q)));
		db.updateQueueItem(id, updates);
	}

	function removeQueueItem(id: string) {
		setQueueItems((prev) => prev.filter((q) => q.id !== id));
		db.deleteQueueItem(id);
	}

	function createPlaylist(name: string) {
		const playlist: Playlist = {
			id: crypto.randomUUID(),
			name,
			color: PLAYLIST_COLORS[playlists.length % PLAYLIST_COLORS.length],
			items: [],
			createdAt: Date.now(),
		};
		setPlaylists((prev) => [...prev, playlist]);
		db.insertPlaylist(playlist);
	}

	function createAndAddToPlaylist(name: string, item: QueueItem) {
		const playlist: Playlist = {
			id: crypto.randomUUID(),
			name,
			color: PLAYLIST_COLORS[playlists.length % PLAYLIST_COLORS.length],
			items: [item],
			createdAt: Date.now(),
		};
		setPlaylists((prev) => [...prev, playlist]);
		const waitForItem = pendingInserts.current.get(item.id) ?? Promise.resolve();
		db.insertPlaylist(playlist)
			.then(() => waitForItem)
			.then(() => db.addItemToPlaylist(playlist.id, item.id, 0));
	}

	function renamePlaylist(id: string, name: string) {
		setPlaylists((prev) => prev.map((p) => (p.id === id ? { ...p, name } : p)));
		db.updatePlaylist(id, { name });
	}

	function changePlaylistColor(id: string, color: string) {
		setPlaylists((prev) => prev.map((p) => (p.id === id ? { ...p, color } : p)));
		db.updatePlaylist(id, { color });
	}

	function deletePlaylist(id: string) {
		setPlaylists((prev) => prev.filter((p) => p.id !== id));
		db.deletePlaylist(id);
	}

	function addToPlaylist(playlistId: string, item: QueueItem) {
		setPlaylists((prev) =>
			prev.map((p) => {
				if (p.id !== playlistId) return p;
				if (p.items.some((i) => i.id === item.id)) return p;
				return { ...p, items: [...p.items, item] };
			})
		);
		const position = playlists.find((p) => p.id === playlistId)?.items.length ?? 0;
		const waitForItem = pendingInserts.current.get(item.id) ?? Promise.resolve();
		waitForItem.then(() => db.addItemToPlaylist(playlistId, item.id, position));
	}

	function removeFromPlaylist(playlistId: string, itemId: string) {
		setPlaylists((prev) =>
			prev.map((p) => (p.id === playlistId ? { ...p, items: p.items.filter((i) => i.id !== itemId) } : p))
		);
		db.removeItemFromPlaylist(playlistId, itemId);
	}

	const pendingCount = queueItems.filter((q) => q.status === "pending" || q.status === "processing").length;

	if (loading) {
		return (
			<div className="max-w-7xl mx-auto min-h-screen flex flex-col w-full">
				<DashboardHeader />
				<div className="flex-1 flex items-center justify-center">
					<p className="text-white/30 text-sm">Loading your library…</p>
				</div>
			</div>
		);
	}

	return (
		<div className="max-w-7xl mx-auto min-h-screen flex flex-col w-full">
			<DashboardHeader />
			<DashboardTabs activeTab={activeTab} onTabChange={setActiveTab} queueBadge={pendingCount} />

			<main id="main" className="flex-1 px-4 sm:px-8 lg:px-12 py-8">
				<AnimatePresence mode="wait">
					<div key={activeTab}>
						{activeTab === "library" && (
							<TrackLibrary tracks={tracks} onAdd={addTrack} onUpdate={updateTrack} onRemove={removeTrack} />
						)}
						{activeTab === "builder" && (
							<TransitionBuilder tracks={tracks} onTransitionComplete={addQueueItem} onUpdateQueue={updateQueueItem} />
						)}
						{activeTab === "queue" && (
							<GenerationQueue items={queueItems} playlists={playlists} onAddToPlaylist={addToPlaylist} onRemove={removeQueueItem} onCreatePlaylist={createPlaylist} onCreateAndAddToPlaylist={createAndAddToPlaylist} />
						)}
						{activeTab === "playlists" && (
							<PlaylistManager
								playlists={playlists}
								onCreate={createPlaylist}
								onRename={renamePlaylist}
								onChangeColor={changePlaylistColor}
								onDelete={deletePlaylist}
								onRemoveItem={removeFromPlaylist}
							/>
						)}
					</div>
				</AnimatePresence>
			</main>
		</div>
	);
}
