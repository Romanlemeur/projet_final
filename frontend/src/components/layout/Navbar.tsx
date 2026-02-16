import Link from "next/link";
import { NAV_LINKS } from "@/lib/constants";

export default function Navbar() {
	return (
		<>
			<header className="fixed top-6 left-0 right-0 z-50 flex justify-center px-4">
				<nav className="bg-black/40 backdrop-blur-xl border border-white/10 rounded-full px-6 py-3 flex items-center justify-between gap-12 shadow-2xl shadow-black/50" aria-label="Main navigation">
					<Link href="/" className="flex items-center gap-2 group">
						<span className="font-bold tracking-tight text-lg">Project</span>
					</Link>

					<div className="hidden md:flex items-center gap-8">
						{NAV_LINKS.map((link) => (
							<a key={link.href} href={link.href} className="text-sm font-medium text-white/60 hover:text-white transition-colors">
								{link.label}
							</a>
						))}
					</div>

					<div className="flex items-center gap-4">
						<Link href="/login" className="text-sm font-medium text-white hover:text-green-400 transition-colors hidden sm:block">
							Log in
						</Link>
						<Link href="/signup" className="bg-white text-black text-sm font-bold px-5 py-2 rounded-full hover:scale-105 transition-transform">
							Get started
						</Link>
					</div>
				</nav>
			</header>
		</>
	);
}
