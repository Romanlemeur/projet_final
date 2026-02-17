import Navbar from "@/components/layout/Navbar";
import Hero from "@/components/sections/Hero";
import LogoStrip from "@/components/sections/LogoStrip";
import Footer from "@/components/layout/Footer";

export default function Home() {
	return (
		<>
			<Navbar />
			<main id="main">
				<Hero />
				<LogoStrip />
			</main>
			<Footer />
		</>
	);
}
