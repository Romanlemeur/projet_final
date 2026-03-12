"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { createClient } from "@/lib/supabase/client";
import { stringToColor } from "@/lib/dashboard-data";
import { AnimatePresence, motion } from "framer-motion";

function UserAvatar() {
	const [initials, setInitials] = useState("");
	const [color, setColor] = useState("#1DB954");
	const [open, setOpen] = useState(false);
	const menuRef = useRef<HTMLDivElement>(null);
	const router = useRouter();

	useEffect(() => {
		const supabase = createClient();
		supabase.auth.getUser().then(({ data: { user } }) => {
			if (!user) return;
			const name = user.user_metadata?.full_name ?? user.email ?? "";
			setInitials(name.trim().charAt(0).toUpperCase() || "?");
			setColor(stringToColor(user.email ?? name));
		});
	}, []);

	useEffect(() => {
		function handleClick(e: MouseEvent) {
			if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
				setOpen(false);
			}
		}
		if (open) document.addEventListener("mousedown", handleClick);
		return () => document.removeEventListener("mousedown", handleClick);
	}, [open]);

	async function handleSignOut() {
		const supabase = createClient();
		await supabase.auth.signOut();
		router.push("/");
	}

	return (
		<div ref={menuRef} className="relative">
			<button
				type="button"
				onClick={() => setOpen((o) => !o)}
				className="w-10 h-10 rounded-full flex items-center justify-center font-bold text-sm text-black shrink-0 hover:opacity-90 transition-opacity"
				style={{ backgroundColor: color }}
				aria-label="Account menu"
			>
				{initials || "?"}
			</button>

			<AnimatePresence>
				{open && (
					<motion.div
						initial={{ opacity: 0, y: -6, scale: 0.97 }}
						animate={{ opacity: 1, y: 0, scale: 1 }}
						exit={{ opacity: 0, y: -6, scale: 0.97 }}
						transition={{ duration: 0.12 }}
						className="absolute right-0 top-full mt-2 w-44 rounded-xl bg-[#1a1a1a] border border-white/10 shadow-2xl z-50 overflow-hidden py-1"
					>
						<Link
							href="/"
							onClick={() => setOpen(false)}
							className="flex items-center gap-3 px-4 py-2.5 text-sm text-white/70 hover:text-white hover:bg-white/[0.06] transition-colors"
						>
							<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
								<path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
								<polyline points="9 22 9 12 15 12 15 22" />
							</svg>
							Home
						</Link>

						<div className="h-px bg-white/[0.06] mx-2 my-1" />

						<button
							type="button"
							onClick={handleSignOut}
							className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-red-400 hover:text-red-300 hover:bg-white/[0.06] transition-colors"
						>
							<svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
								<path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
								<polyline points="16 17 21 12 16 7" />
								<line x1="21" y1="12" x2="9" y2="12" />
							</svg>
							Sign out
						</button>
					</motion.div>
				)}
			</AnimatePresence>
		</div>
	);
}

export default function DashboardHeader() {
	return (
		<header className="flex items-center justify-between px-4 sm:px-8 lg:px-12 py-5">
			<Link href="/" className="flex items-center gap-2.5 group rounded-lg">
				<span className="font-bold text-xl tracking-tight text-white">btekol</span>
			</Link>

			<UserAvatar />
		</header>
	);
}
