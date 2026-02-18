import Link from "next/link";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
	return (
		<div className="min-h-screen bg-[#050505] flex flex-col items-center justify-center px-4 py-12 relative overflow-hidden">
			<Link href="/" className="flex items-center gap-2 mb-8 relative z-10 group rounded-lg">
				<span className="font-bold tracking-tight text-lg text-white">Project</span>
			</Link>

			<div className="relative z-10 w-full flex justify-center">{children}</div>
		</div>
	);
}
