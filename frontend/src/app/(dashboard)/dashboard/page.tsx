import type { Metadata } from "next";
import DashboardClient from "@/components/dashboard/DashboardClient";

export const metadata: Metadata = {
	title: "Dashboard - Project",
	description: "Upload tracks, build transitions, and manage your generation queue.",
};

export default function DashboardPage() {
	return <DashboardClient />;
}
