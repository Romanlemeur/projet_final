import Image from "next/image";
import Link from "next/link";
import AnimateOnScroll from "@/components/ui/AnimateOnScroll";

export default function Hero() {
	return (
		<section className="min-h-screen flex flex-col justify-center items-center px-4 pt-44 pb-32 relative overflow-hidden">
			<div className="max-w-7xl mx-auto w-full z-10 grid grid-cols-1 lg:grid-cols-2 gap-12 lg:gap-6 items-center">
				<AnimateOnScroll delay={0.3} className="flex justify-center lg:justify-center order-2 lg:order-1">
					<div className="relative w-[280px] sm:w-[340px] lg:w-[390px] aspect-square">
						<div className="absolute -inset-12 bg-green-500/15 blur-[80px] rounded-full" aria-hidden="true" />
						<Image src="/disc.png" alt="Vinyl disc" width={390} height={390} priority className="relative z-10 drop-shadow-2xl animate-float" />
					</div>
				</AnimateOnScroll>

				<div className="text-center lg:text-left order-1 lg:order-2">
					<AnimateOnScroll>
						<h1 className="text-5xl sm:text-6xl md:text-7xl lg:text-8xl font-extrabold tracking-tighter leading-[0.9] mb-8">
							<span className="text-transparent bg-clip-text bg-gradient-to-br from-white via-white to-white/40">Project</span>
						</h1>
					</AnimateOnScroll>

					<AnimateOnScroll delay={0.2}>
						<p className="text-base sm:text-xl text-white/60 max-w-2xl mx-auto lg:mx-0 mb-12 leading-relaxed font-medium">
							Seamless, beat-matched transitions for your playlists. <br className="hidden sm:block" />
							Powered by AI, perfected by you.
						</p>
					</AnimateOnScroll>

					<AnimateOnScroll delay={0.3}>
						<div className="flex flex-col sm:flex-row items-center justify-center lg:justify-start gap-4">
							<Link href="/signup" className="w-full sm:w-auto bg-green-500 text-black font-semibold text-base px-7 py-3.5 rounded-full hover:scale-105 hover:bg-green-400 transition-all shadow-[0_0_40px_-10px_rgba(29,185,84,0.5)]">
								Start Mixing Free
							</Link>
							<a href="#how" className="w-full sm:w-auto px-7 py-3.5 rounded-full font-semibold text-base border border-white/10 hover:bg-white/5 transition-colors">
								View Demo
							</a>
						</div>
					</AnimateOnScroll>
				</div>
			</div>
		</section>
	);
}
