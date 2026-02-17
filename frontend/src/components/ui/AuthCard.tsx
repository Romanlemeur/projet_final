interface AuthCardProps {
	children: React.ReactNode;
}

export default function AuthCard({ children }: AuthCardProps) {
	return <div className="bento-card max-w-md w-full p-8 rounded-3xl">{children}</div>;
}
