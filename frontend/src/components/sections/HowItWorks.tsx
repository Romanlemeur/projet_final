import Image from "next/image";
import { STEPS } from "@/lib/constants";

export default function HowItWorks() {
	return (
		<section id="how" className="py-32 px-4 relative">
			<div className="absolute inset-0 pointer-events-none" aria-hidden="true">
				<div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-transparent to-white/[0.02]" />
				<div className="absolute inset-x-0 top-32 bottom-32 bg-white/[0.02]" />
				<div className="absolute inset-x-0 bottom-0 h-32 bg-gradient-to-t from-transparent to-white/[0.02]" />
			</div>
			<div className="max-w-7xl mx-auto relative">
				<div className="grid md:grid-cols-2 gap-16 items-center">
					<div>
						<span className="text-green-500 font-bold uppercase tracking-widest text-sm mb-4 block">Workflow</span>
						<h2 className="text-4xl md:text-6xl font-extrabold tracking-tighter mb-8">
							3 Steps to <br />
							Pro Sound.
						</h2>
						<div className="space-y-8">
							{STEPS.map((step) => (
								<div key={step.number} className="flex gap-6 items-start group">
									<div className="w-12 h-12 rounded-full border border-white/10 bg-white/5 flex items-center justify-center text-xl font-bold text-white group-hover:bg-green-500 group-hover:text-black transition-colors shrink-0">{step.number}</div>
									<div>
										<h4 className="text-xl font-bold mb-2">{step.title}</h4>
										<p className="text-white/60">{step.description}</p>
									</div>
								</div>
							))}
						</div>
					</div>

					<div>
						<div className="rounded-[40px] border border-white/10 bg-gradient-to-b from-white/5 to-transparent relative overflow-hidden flex flex-col p-6 gap-4">
							<div className="rounded-2xl overflow-hidden border border-white/5">
								<Image src="/transition-builder.png" alt="Transition Builder - select tracks and configure AI transition" width={1340} height={878} className="w-full h-auto" />
							</div>

							<div className="h-16 w-full bg-green-500/20 rounded-2xl border border-green-500/30 flex items-center justify-center shrink-0">
								<span className="text-green-500 font-mono text-xs uppercase tracking-widest animate-bounce">Processing...</span>
							</div>

							<div className="rounded-2xl overflow-hidden border border-white/5">
								<Image src="/generation-queue.png" alt="Generation Queue - completed AI transition with waveform" width={1340} height={480} className="w-full h-auto" />
							</div>
						</div>
					</div>
				</div>
			</div>
		</section>
	);
}
