"use client";

import { useState } from "react";
import { AnimatePresence } from "framer-motion";
import DashboardHeader from "./DashboardHeader";
import DashboardTabs, { type TabId } from "./DashboardTabs";
import TrackLibrary from "./TrackLibrary";

const TAB_PANELS: Record<TabId, React.ReactNode> = {
	library: <TrackLibrary />,
};

export default function DashboardClient() {
	const [activeTab, setActiveTab] = useState<TabId>("library");

	return (
		<div className="max-w-7xl mx-auto min-h-screen flex flex-col w-full">
			<DashboardHeader />
			<DashboardTabs activeTab={activeTab} onTabChange={setActiveTab} />

			<main id="main" className="flex-1 px-4 sm:px-8 lg:px-12 py-8">
				<AnimatePresence mode="wait">
					<div key={activeTab}>{TAB_PANELS[activeTab]}</div>
				</AnimatePresence>
			</main>
		</div>
	);
}
