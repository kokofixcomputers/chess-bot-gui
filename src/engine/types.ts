export type Color = 'w' | 'b';
export type PieceType = 'K' | 'Q' | 'R' | 'B' | 'N' | 'P';
export type Piece = `${Color}${PieceType}`;

export type Square = [number, number]; // [row, col] 0-indexed, row0=rank8

export interface CastleRights {
  wK: boolean; // white kingside
  wQ: boolean; // white queenside
  bK: boolean;
  bQ: boolean;
}

export interface GameState {
  board: (Piece | null)[][];
  turn: Color;
  castleRights: CastleRights;
  enPassant: Square | null; // target square
  halfMoveClock: number;
  fullMoveNumber: number;
}

export interface Move {
  from: Square;
  to: Square;
  piece: Piece;
  captured?: Piece;
  promotion?: PieceType;
  castling?: 'K' | 'Q'; // kingside or queenside
  enPassant?: boolean;
}

export type GameResult = 'ongoing' | 'checkmate' | 'stalemate' | 'draw-50' | 'draw-repetition';
