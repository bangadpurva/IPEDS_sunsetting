import type { Metadata } from 'next';
import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import Link from 'next/link';

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
});

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
});

export const metadata: Metadata = {
  title: 'Viascope — See where your interests could take you',
  description: 'Explore degrees, careers, and colleges using transparent IPEDS and BLS evidence.',
  openGraph: {
    title: 'Viascope — See where your interests could take you',
    description: 'Explore degrees, careers, and colleges using transparent IPEDS and BLS evidence.',
    images: [{ url: '/og.png', width: 1200, height: 630, alt: 'Viascope student decision workspace' }],
  },
  twitter: { card: 'summary_large_image', images: ['/og.png'] },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        {children}
        <aside className="guide-notice" role="note" aria-label="Viascope guidance disclaimer"><b>Use Viascope as a guide.</b><span>Results reflect available public data and may include AI-assisted explanations. They are not admissions, financial, or career guarantees.</span><Link href="/methodology">Review the evidence and limits</Link></aside>
      </body>
    </html>
  );
}
