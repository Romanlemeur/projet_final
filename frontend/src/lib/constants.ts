export const NAV_LINKS = [
	{ label: "How it works", href: "#how" },
	{ label: "Pricing", href: "#pricing" },
] as const;

export const FOOTER_LINKS = [
	{ label: "Twitter", href: "#" },
	{ label: "Instagram", href: "#" },
	{ label: "Privacy", href: "#" },
] as const;

export const SERVICES = ["Spotify", "Apple Music", "SoundCloud", "Tidal", "YouTube Music", "Deezer", "Amazon Music", "Pandora"] as const;

export const STEPS = [
	{
		number: 1,
		title: "Upload Tracks",
		description: "Drag and drop your audio files. We support all major formats.",
	},
	{
		number: 2,
		title: "Auto-Generate",
		description: "Our AI analyzes the tracks and builds the perfect transition bridge.",
	},
	{
		number: 3,
		title: "Export",
		description: "Download your seamless mix or sync back to your library.",
	},
] as const;
