import Navbar from "@/components/layout/Navbar";
import Footer from "@/components/layout/Footer";
import Hero from "@/components/sections/Hero";

export default function Home() {
	return (
		<>
			<Navbar />
			<main id="main">
				<Hero />
			</main>
			<Footer />
		</>
	);
}
