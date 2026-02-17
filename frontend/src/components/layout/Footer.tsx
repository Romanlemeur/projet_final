import { FOOTER_LINKS } from "@/lib/constants";

export default function Footer() {
	return (
		<footer className="py-12 border-t border-white/5 mt-20">
			<div className="max-w-7xl mx-auto px-4 flex flex-col md:flex-row justify-between items-center gap-8">
				<div className="flex items-center gap-2">
					<span className="font-bold">Project</span>
				</div>
				<div className="text-white/40 text-sm">&copy; 2026 Project Inc. All rights reserved.</div>
				<div className="flex gap-6">
					{FOOTER_LINKS.map((link) => (
						<a key={link.label} href={link.href} className="text-white/40 hover:text-white transition-colors">
							{link.label}
						</a>
					))}
				</div>
			</div>
		</footer>
	);
}
