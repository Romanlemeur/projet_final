"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface AnimateOnScrollProps {
	children: ReactNode;
	className?: string;
	delay?: number;
}

export default function AnimateOnScroll({ children, className = "", delay = 0 }: AnimateOnScrollProps) {
	return (
		<motion.div
			initial={{ opacity: 0, y: 30, scale: 0.98 }}
			whileInView={{ opacity: 1, y: 0, scale: 1 }}
			viewport={{ once: true, amount: 0.1 }}
			transition={{
				duration: 0.8,
				delay,
				ease: [0.16, 1, 0.3, 1],
			}}
			className={className}
		>
			{children}
		</motion.div>
	);
}
