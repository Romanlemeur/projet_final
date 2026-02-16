"use client";

import { motion } from "framer-motion";
import { SERVICES } from "@/lib/constants";

export default function LogoStrip() {
	return (
		<section className="py-14 border-y border-white/5 relative w-full overflow-x-clip">
			<div className="absolute left-0 top-0 bottom-0 w-40 bg-gradient-to-r from-[#050505] to-transparent z-10 pointer-events-none" />
			<div className="absolute right-0 top-0 bottom-0 w-40 bg-gradient-to-l from-[#050505] to-transparent z-10 pointer-events-none" />

			<motion.div
				className="flex gap-14 items-center w-max"
				animate={{ x: ["0%", "-50%"] }}
				transition={{
					x: {
						repeat: Infinity,
						repeatType: "loop",
						duration: 25,
						ease: "linear",
					},
				}}
			>
				{[...SERVICES, ...SERVICES].map((name, i) => (
					<span key={`${name}-${i}`} className="text-lg font-semibold text-white/25 shrink-0">
						{name}
					</span>
				))}
			</motion.div>
		</section>
	);
}
