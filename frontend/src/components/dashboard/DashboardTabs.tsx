"use client";

import { motion } from "framer-motion";
import { mockQueue } from "@/lib/dashboard-data";

export type TabId = "library" | "builder" | "queue";

interface Tab {
	id: TabId;
	label: string;
	icon: React.ReactNode;
	badge?: number;
}

function LibraryIcon() {
	return (
		<svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<path d="M9 18V5l12-2v13" />
			<circle cx="6" cy="18" r="3" />
			<circle cx="18" cy="16" r="3" />
		</svg>
	);
}

function BuilderIcon() {
	return (
		<svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<polyline points="16 3 21 3 21 8" />
			<line x1="4" y1="20" x2="21" y2="3" />
			<polyline points="21 16 21 21 16 21" />
			<line x1="15" y1="15" x2="21" y2="21" />
			<line x1="4" y1="4" x2="9" y2="9" />
		</svg>
	);
}

function QueueIcon() {
	return (
		<svg className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
			<line x1="8" y1="6" x2="21" y2="6" />
			<line x1="8" y1="12" x2="21" y2="12" />
			<line x1="8" y1="18" x2="21" y2="18" />
			<line x1="3" y1="6" x2="3.01" y2="6" />
			<line x1="3" y1="12" x2="3.01" y2="12" />
			<line x1="3" y1="18" x2="3.01" y2="18" />
		</svg>
	);
}

const pendingCount = mockQueue.filter((q) => q.status !== "completed").length;

const tabs: Tab[] = [
	{ id: "library", label: "Track Library", icon: <LibraryIcon /> },
	{ id: "builder", label: "Transition Builder", icon: <BuilderIcon /> },
	{
		id: "queue",
		label: "Generation Queue",
		icon: <QueueIcon />,
		badge: pendingCount > 0 ? pendingCount : undefined,
	},
];

interface DashboardTabsProps {
	activeTab: TabId;
	onTabChange: (tab: TabId) => void;
}

export default function DashboardTabs({ activeTab, onTabChange }: DashboardTabsProps) {
	return (
		<nav className="flex gap-2 px-4 sm:px-8 lg:px-12 border-b border-white/5">
			{tabs.map((tab) => (
				<button key={tab.id} onClick={() => onTabChange(tab.id)} className={`relative flex items-center gap-2.5 px-5 py-4 text-base font-medium transition-colors rounded-t-lg ${activeTab === tab.id ? "text-white" : "text-white/40 hover:text-white/70"}`}>
					{tab.icon}
					<span className="hidden sm:inline">{tab.label}</span>
					{tab.badge !== undefined && <span className="ml-1 bg-green-500 text-black text-[10px] font-bold rounded-full w-4.5 h-4.5 flex items-center justify-center leading-none">{tab.badge}</span>}
					{activeTab === tab.id && <motion.div layoutId="tab-underline" className="absolute bottom-0 left-0 right-0 h-0.5 bg-green-500" transition={{ type: "spring", stiffness: 500, damping: 35 }} />}
				</button>
			))}
		</nav>
	);
}
