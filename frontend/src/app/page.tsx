import Navbar from "@/components/layout/Navbar";
import Hero from "@/components/sections/Hero";
import LogoStrip from "@/components/sections/LogoStrip";
import HowItWorks from "@/components/sections/HowItWorks";
import Pricing from "@/components/sections/Pricing";
import Footer from "@/components/layout/Footer";

export default function Home() {
	return (
		<>
			<Navbar />
			<main id="main">
				<Hero />
				<LogoStrip />
				<HowItWorks />
				<div className="max-w-6xl mx-auto px-4">
					<hr className="border-white/5" />
				</div>
				<Pricing />
			</main>
			<Footer />
		</>
	);
}
