import React from 'react';
import { ChessPiece } from './ChessPiece';

interface Props {
  color: 'w' | 'b';
  onSelect: (piece: 'Q' | 'R' | 'B' | 'N') => void;
}

const OPTIONS: Array<'Q' | 'R' | 'B' | 'N'> = ['Q', 'R', 'B', 'N'];

export const PromotionModal: React.FC<Props> = ({ color, onSelect }) => {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(6px)' }}>
      <div style={{
        background: 'linear-gradient(135deg, #1a1d2e, #13152a)',
        border: '1px solid rgba(255,255,255,0.1)',
        borderRadius: 20,
        padding: 28,
        boxShadow: '0 32px 80px rgba(0,0,0,0.6)',
      }}>
        <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: 12, fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase', marginBottom: 20, textAlign: 'center' }}>
          Choose Promotion
        </div>
        <div style={{ display: 'flex', gap: 12 }}>
          {OPTIONS.map(pt => (
            <button
              key={pt}
              onClick={() => onSelect(pt)}
              style={{
                width: 76,
                height: 90,
                borderRadius: 14,
                background: 'rgba(255,255,255,0.05)',
                border: '1px solid rgba(255,255,255,0.1)',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 8,
                transition: 'all 0.15s',
              }}
              onMouseEnter={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(99,102,241,0.2)';
                (e.currentTarget as HTMLElement).style.borderColor = 'rgba(99,102,241,0.5)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(-2px)';
              }}
              onMouseLeave={e => {
                (e.currentTarget as HTMLElement).style.background = 'rgba(255,255,255,0.05)';
                (e.currentTarget as HTMLElement).style.borderColor = 'rgba(255,255,255,0.1)';
                (e.currentTarget as HTMLElement).style.transform = 'translateY(0)';
              }}
            >
              <ChessPiece piece={`${color}${pt}`} size={48} />
              <span style={{ color: 'rgba(255,255,255,0.4)', fontSize: 11, fontWeight: 600 }}>{pt}</span>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
