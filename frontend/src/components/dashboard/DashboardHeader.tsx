import Link from "next/link";

function UserAvatar() {
	return (
		<div className="w-10 h-10 rounded-full bg-white/10 border border-white/10 flex items-center justify-center">
			<svg className="w-5 h-5 text-white/60" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
				<path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z" />
			</svg>
		</div>
	);
}

export default function DashboardHeader() {
	return (
		<header className="flex items-center justify-between px-4 sm:px-8 lg:px-12 py-5">
			<Link href="/" className="flex items-center gap-2.5 group rounded-lg">
				<span className="font-bold text-xl tracking-tight text-white">Project</span>
			</Link>

			<UserAvatar />
		</header>
	);
}
