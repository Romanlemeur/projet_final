import AnimateOnScroll from "@/components/ui/AnimateOnScroll";
import { PRICING_TIERS } from "@/lib/constants";

function CheckMark() {
	return (
		<svg className="w-4 h-4 text-green-500 shrink-0" fill="none" stroke="currentColor" strokeWidth={2.5} viewBox="0 0 24 24" aria-hidden="true">
			<polyline points="20 6 9 17 4 12" />
		</svg>
	);
}

function XMark() {
	return (
		<svg className="w-4 h-4 text-white/20 shrink-0" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24" aria-hidden="true">
			<path d="M18 6L6 18M6 6l12 12" />
		</svg>
	);
}

export default function Pricing() {
	return (
		<section id="pricing" className="py-32 px-4">
			<div className="max-w-6xl mx-auto">
				<div className="grid md:grid-cols-3 gap-6">
					{PRICING_TIERS.map((tier, i) => (
						<AnimateOnScroll key={tier.name} delay={i * 0.1}>
							<div className={`rounded-[32px] h-full flex flex-col relative ${tier.highlighted ? "bg-gradient-to-b from-green-500/20 via-green-500/5 to-transparent border border-green-500/20 p-8" : "bento-card p-8"}`}>
								{"badge" in tier && tier.badge && <div className="absolute top-4 right-4 bg-green-500 text-black text-[10px] uppercase font-bold px-3 py-1 rounded-full">{tier.badge}</div>}
								<h3 className={`text-lg font-bold mb-1 ${tier.highlighted ? "text-green-500" : "text-white/80"}`}>{tier.name}</h3>
								<div className="flex items-baseline gap-1 mb-2">
									<span className="text-4xl font-extrabold">{tier.price}</span>
									{tier.period && <span className="text-base font-medium text-white/40">{tier.period}</span>}
								</div>
								<p className="text-sm text-white/50 mb-6 leading-relaxed">{tier.description}</p>

								<a href="#" className={`block py-3.5 rounded-full font-bold text-center transition-all mb-8 ${tier.highlighted ? "bg-green-500 text-black hover:scale-105 hover:bg-green-400" : "border border-white/10 hover:bg-white/10 text-white"}`}>
									{tier.cta}
								</a>

								<div className="border-t border-white/5 pt-6">
									<p className="text-xs font-semibold uppercase tracking-wider text-white/40 mb-4">What&apos;s included</p>
									<ul className="space-y-3">
										{tier.features.map((feature) => (
											<li key={feature.text} className="flex items-center gap-3">
												{feature.included ? <CheckMark /> : <XMark />}
												<span className={`text-sm ${feature.included ? "text-white/80" : "text-white/30 line-through"}`}>{feature.text}</span>
											</li>
										))}
									</ul>
								</div>
							</div>
						</AnimateOnScroll>
					))}
				</div>
			</div>
		</section>
	);
}
