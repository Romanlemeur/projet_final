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

export const PRICING_TIERS = [
	{
		name: "Free",
		price: "$0",
		period: null,
		description: "For casual listeners who want to try AI mixing.",
		cta: "Start Free",
		highlighted: false,
		features: [
			{ text: "5 mixes per month", included: true },
			{ text: "Basic transition styles", included: true },
			{ text: "MP3 export", included: true },
			{ text: "Lossless export (WAV/FLAC)", included: false },
			{ text: "Harmonic key detection", included: false },
			{ text: "Energy curve control", included: false },
			{ text: "Custom transition duration", included: false },
			{ text: "Streaming service import", included: false },
		],
	},
	{
		name: "Pro",
		price: "$12",
		period: "/mo",
		description: "For DJs and creators who need the full toolkit.",
		cta: "Get Pro",
		highlighted: true,
		badge: "Most Popular",
		features: [
			{ text: "Unlimited mixes", included: true },
			{ text: "All transition styles", included: true },
			{ text: "MP3, WAV & FLAC export", included: true },
			{ text: "Lossless export", included: true },
			{ text: "Harmonic key detection", included: true },
			{ text: "Energy curve control", included: true },
			{ text: "Custom transition duration", included: true },
			{ text: "Streaming service import", included: true },
			{ text: "API access", included: false },
			{ text: "Team collaboration", included: false },
		],
	},
	{
		name: "Studio",
		price: "$29",
		period: "/mo",
		description: "For teams and professionals at scale.",
		cta: "Contact Sales",
		highlighted: false,
		features: [
			{ text: "Everything in Pro", included: true },
			{ text: "Unlimited mixes", included: true },
			{ text: "All export formats", included: true },
			{ text: "Lossless export", included: true },
			{ text: "Full AI engine access", included: true },
			{ text: "Streaming service import", included: true },
			{ text: "API access", included: true },
			{ text: "Team collaboration", included: true },
			{ text: "Priority support", included: true },
			{ text: "Custom integrations", included: true },
		],
	},
] as const;
