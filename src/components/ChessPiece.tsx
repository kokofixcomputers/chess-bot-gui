import React from 'react';

interface Props {
  piece: string;
  size?: number;
  lifted?: boolean;
}

export const ChessPiece: React.FC<Props> = ({ piece, size = 52, lifted = false }) => {
  const isWhite = piece[0] === 'w';
  const type = piece[1];

  const fill = isWhite ? '#dde6f5' : '#1e2235';
  const stroke = isWhite ? '#94a3b8' : '#0d0f1a';
  const accent = isWhite ? 'rgba(0,0,0,0.18)' : 'rgba(255,255,255,0.15)';
  const highlight = isWhite ? 'rgba(255,255,255,0.6)' : 'rgba(255,255,255,0.1)';

  const shapes: Record<string, JSX.Element> = {
    K: (
      <g>
        <rect x="17" y="4" width="6" height="10" rx="2" fill={fill} stroke={stroke} strokeWidth="1"/>
        <rect x="13" y="7" width="14" height="5" rx="2" fill={fill} stroke={stroke} strokeWidth="1"/>
        <path d="M9 37 L11 21 Q20 15 29 21 L31 37 Q20 43 9 37Z" fill={fill} stroke={stroke} strokeWidth="1.2"/>
        <path d="M12 37 Q20 41 28 37" stroke={accent} strokeWidth="1.5" fill="none"/>
        <ellipse cx="20" cy="37.5" rx="11" ry="3.5" fill={accent} opacity="0.5"/>
        <rect x="17" y="4" width="6" height="4" rx="1" fill={highlight} opacity="0.6"/>
      </g>
    ),
    Q: (
      <g>
        <circle cx="20" cy="8" r="3.5" fill={fill} stroke={stroke} strokeWidth="1"/>
        <circle cx="9" cy="13" r="3" fill={fill} stroke={stroke} strokeWidth="1"/>
        <circle cx="31" cy="13" r="3" fill={fill} stroke={stroke} strokeWidth="1"/>
        <path d="M7 37 L9 18 L13 27 L20 12 L27 27 L31 18 L33 37 Q20 43 7 37Z" fill={fill} stroke={stroke} strokeWidth="1.2"/>
        <ellipse cx="20" cy="37.5" rx="13" ry="3.5" fill={accent} opacity="0.5"/>
        <circle cx="20" cy="8" r="1.5" fill={highlight}/>
      </g>
    ),
    R: (
      <g>
        <rect x="11" y="7" width="5" height="9" rx="1.5" fill={fill} stroke={stroke} strokeWidth="1"/>
        <rect x="18" y="7" width="4" height="9" rx="1.5" fill={fill} stroke={stroke} strokeWidth="1"/>
        <rect x="24" y="7" width="5" height="9" rx="1.5" fill={fill} stroke={stroke} strokeWidth="1"/>
        <rect x="13" y="14" width="14" height="3" fill={fill}/>
        <path d="M12 37 L13 17 L27 17 L28 37 Q20 42 12 37Z" fill={fill} stroke={stroke} strokeWidth="1.2"/>
        <ellipse cx="20" cy="37.5" rx="10" ry="3.5" fill={accent} opacity="0.5"/>
        <rect x="14" y="8" width="4" height="3" rx="1" fill={highlight} opacity="0.5"/>
      </g>
    ),
    B: (
      <g>
        <circle cx="20" cy="8" r="4.5" fill={fill} stroke={stroke} strokeWidth="1"/>
        <path d="M14 37 Q15 17 20 13 Q25 17 26 37 Q20 42 14 37Z" fill={fill} stroke={stroke} strokeWidth="1.2"/>
        <circle cx="20" cy="8" r="2" fill={accent}/>
        <line x1="17" y1="35" x2="23" y2="35" stroke={accent} strokeWidth="1.5"/>
        <ellipse cx="20" cy="37" rx="9" ry="3.5" fill={accent} opacity="0.5"/>
        <circle cx="20" cy="7" r="1.2" fill={highlight}/>
      </g>
    ),
    N: (
      <g>
        <path d="M12 37 L13 23 Q10 16 13 10 Q16 6 21 7 Q28 8 27 15 Q26 19 22 21 L27 37 Q20 42 12 37Z" fill={fill} stroke={stroke} strokeWidth="1.2"/>
        <circle cx="17.5" cy="12" r="2" fill={accent}/>
        <path d="M14 19 Q17 17 20 19" stroke={accent} strokeWidth="1.2" fill="none"/>
        <ellipse cx="20" cy="37" rx="9" ry="3.5" fill={accent} opacity="0.5"/>
        <ellipse cx="18" cy="10" rx="3" ry="2" fill={highlight} opacity="0.4" transform="rotate(-15 18 10)"/>
      </g>
    ),
    P: (
      <g>
        <circle cx="20" cy="12" r="6.5" fill={fill} stroke={stroke} strokeWidth="1"/>
        <path d="M14 37 L15 23 Q20 18 25 23 L26 37 Q20 42 14 37Z" fill={fill} stroke={stroke} strokeWidth="1.2"/>
        <ellipse cx="20" cy="37" rx="9" ry="3.5" fill={accent} opacity="0.5"/>
        <circle cx="18" cy="10" r="2.5" fill={highlight} opacity="0.5"/>
      </g>
    ),
  };

  const shadow = lifted
    ? (isWhite ? 'drop-shadow(0 8px 16px rgba(0,0,0,0.4))' : 'drop-shadow(0 8px 20px rgba(0,0,0,0.6))')
    : (isWhite ? 'drop-shadow(0 2px 5px rgba(0,0,0,0.2))' : 'drop-shadow(0 3px 7px rgba(0,0,0,0.5))');

  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 40 44"
      style={{ filter: shadow, transition: 'filter 0.15s, transform 0.15s' }}
    >
      {shapes[type]}
    </svg>
  );
};
