"use client";

import { motion } from "framer-motion";
import { mockTracks } from "@/lib/dashboard-data";

function UploadIcon() {
	return (
		<svg className="w-8 h-8 text-white/30" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
			<polyline points="17 8 12 3 7 8" />
			<line x1="12" y1="3" x2="12" y2="15" />
		</svg>
	);
}

function ClockIcon() {
	return (
		<svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<circle cx="12" cy="12" r="10" />
			<polyline points="12 6 12 12 16 14" />
		</svg>
	);
}

export default function TrackLibrary() {
	return (
		<motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }} className="space-y-8">
			<button type="button" className="w-full border-2 border-dashed border-white/10 rounded-2xl p-10 sm:p-16 flex flex-col items-center gap-4 hover:border-green-500/40 hover:bg-white/[0.02] transition-colors cursor-pointer">
				<UploadIcon />
				<p className="text-white/50 text-base font-medium">
					Drop tracks here or <span className="text-green-500">browse files</span>
				</p>
				<p className="text-white/25 text-sm">MP3, WAV, FLAC - up to 50 MB</p>
			</button>

			<div className="space-y-3">
				{mockTracks.map((track) => (
					<div key={track.id} className="flex items-center gap-5 rounded-xl px-5 py-4 bg-white/[0.03] border border-white/[0.06] hover:border-white/10 transition-colors">
						<div className="w-14 h-14 rounded-lg shrink-0" style={{ backgroundColor: track.color }} />

						<div className="flex-1 min-w-0">
							<p className="text-base font-semibold text-white truncate">{track.title}</p>
							<p className="text-sm text-white/40 truncate">{track.artist}</p>
						</div>

						<div className="hidden sm:flex items-center gap-3 text-sm text-white/40">
							<span className="flex items-center gap-1.5">
								<ClockIcon />
								{track.duration}
							</span>
							<span className="px-2 py-1 rounded bg-white/5 font-mono">{track.bpm} BPM</span>
							<span className="px-2 py-1 rounded bg-white/5 font-mono">{track.key}</span>
						</div>
					</div>
				))}
			</div>
		</motion.div>
	);
}
