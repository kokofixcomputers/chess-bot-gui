import type { Color, Piece, PieceType, Square, GameState, Move, CastleRights } from './types';

// ─── Helpers ─────────────────────────────────────────────────────────────────

export const inBounds = (r: number, c: number) => r >= 0 && r < 8 && c >= 0 && c < 8;
export const pieceColor = (p: Piece | null): Color | null => p ? p[0] as Color : null;
export const pieceType = (p: Piece | null): PieceType | null => p ? p[1] as PieceType : null;
export const makePiece = (color: Color, type: PieceType): Piece => `${color}${type}` as Piece;
export const opponent = (c: Color): Color => c === 'w' ? 'b' : 'w';

export const INITIAL_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

// ─── Board utils ─────────────────────────────────────────────────────────────

export function cloneBoard(board: (Piece | null)[][]): (Piece | null)[][] {
  return board.map(r => [...r]);
}

export function cloneState(state: GameState): GameState {
  return {
    board: cloneBoard(state.board),
    turn: state.turn,
    castleRights: { ...state.castleRights },
    enPassant: state.enPassant ? [...state.enPassant] as Square : null,
    halfMoveClock: state.halfMoveClock,
    fullMoveNumber: state.fullMoveNumber,
  };
}

// ─── FEN ─────────────────────────────────────────────────────────────────────

const FEN_MAP: Record<string, Piece> = {
  K: 'wK', Q: 'wQ', R: 'wR', B: 'wB', N: 'wN', P: 'wP',
  k: 'bK', q: 'bQ', r: 'bR', b: 'bB', n: 'bN', p: 'bP',
};
const PIECE_TO_FEN: Record<Piece, string> = {
  wK: 'K', wQ: 'Q', wR: 'R', wB: 'B', wN: 'N', wP: 'P',
  bK: 'k', bQ: 'q', bR: 'r', bB: 'b', bN: 'n', bP: 'p',
};

export function parseFEN(fen: string): GameState {
  const [piecePlacement, turn, castling, enPassantStr, halfMove, fullMove] = fen.split(' ');
  const board: (Piece | null)[][] = [];
  for (const rank of piecePlacement.split('/')) {
    const row: (Piece | null)[] = [];
    for (const ch of rank) {
      const n = parseInt(ch);
      if (!isNaN(n)) { for (let i = 0; i < n; i++) row.push(null); }
      else row.push(FEN_MAP[ch] ?? null);
    }
    board.push(row);
  }
  const cr: CastleRights = { wK: castling.includes('K'), wQ: castling.includes('Q'), bK: castling.includes('k'), bQ: castling.includes('q') };
  let ep: Square | null = null;
  if (enPassantStr !== '-') {
    const c = enPassantStr.charCodeAt(0) - 97;
    const r = 8 - parseInt(enPassantStr[1]);
    ep = [r, c];
  }
  return { board, turn: turn as Color, castleRights: cr, enPassant: ep, halfMoveClock: parseInt(halfMove) || 0, fullMoveNumber: parseInt(fullMove) || 1 };
}

export function toFEN(state: GameState): string {
  const rows = state.board.map(row => {
    let s = ''; let empty = 0;
    for (const sq of row) {
      if (!sq) { empty++; } else { if (empty) { s += empty; empty = 0; } s += PIECE_TO_FEN[sq]; }
    }
    if (empty) s += empty;
    return s;
  });
  const cr = state.castleRights;
  const castling = [cr.wK ? 'K' : '', cr.wQ ? 'Q' : '', cr.bK ? 'k' : '', cr.bQ ? 'q' : ''].join('') || '-';
  const ep = state.enPassant ? String.fromCharCode(97 + state.enPassant[1]) + (8 - state.enPassant[0]) : '-';
  return `${rows.join('/')} ${state.turn} ${castling} ${ep} ${state.halfMoveClock} ${state.fullMoveNumber}`;
}

// ─── Raw move generation (ignores check) ─────────────────────────────────────

export function getRawMoves(state: GameState, r: number, c: number): Move[] {
  const { board, enPassant } = state;
  const piece = board[r][c];
  if (!piece) return [];
  const side = pieceColor(piece)!;
  const type = pieceType(piece)!;
  const moves: Move[] = [];

  const addMove = (nr: number, nc: number, extra?: Partial<Move>) => {
    if (!inBounds(nr, nc)) return false;
    const target = board[nr][nc];
    if (pieceColor(target) === side) return false;
    moves.push({ from: [r, c], to: [nr, nc], piece, captured: target ?? undefined, ...extra });
    return !target; // can continue sliding if empty
  };

  const slide = (dr: number, dc: number) => {
    let nr = r + dr, nc = c + dc;
    while (inBounds(nr, nc)) {
      const canContinue = addMove(nr, nc);
      if (!canContinue) break;
      nr += dr; nc += dc;
    }
  };

  if (type === 'P') {
    const dir = side === 'w' ? -1 : 1;
    const startRow = side === 'w' ? 6 : 1;
    const promRow = side === 'w' ? 0 : 7;
    // Forward
    if (inBounds(r + dir, c) && !board[r + dir][c]) {
      const toRow = r + dir;
      if (toRow === promRow) {
        for (const pt of ['Q', 'R', 'B', 'N'] as PieceType[]) {
          moves.push({ from: [r, c], to: [toRow, c], piece, promotion: pt });
        }
      } else {
        moves.push({ from: [r, c], to: [toRow, c], piece });
        if (r === startRow && !board[r + 2 * dir][c]) {
          moves.push({ from: [r, c], to: [r + 2 * dir, c], piece });
        }
      }
    }
    // Captures
    for (const dc of [-1, 1]) {
      const nr = r + dir, nc = c + dc;
      if (!inBounds(nr, nc)) continue;
      if (board[nr][nc] && pieceColor(board[nr][nc]) !== side) {
        if (nr === promRow) {
          for (const pt of ['Q', 'R', 'B', 'N'] as PieceType[]) {
            moves.push({ from: [r, c], to: [nr, nc], piece, captured: board[nr][nc]!, promotion: pt });
          }
        } else {
          moves.push({ from: [r, c], to: [nr, nc], piece, captured: board[nr][nc]! });
        }
      }
      // En passant
      if (enPassant && nr === enPassant[0] && nc === enPassant[1]) {
        moves.push({ from: [r, c], to: [nr, nc], piece, captured: makePiece(opponent(side), 'P'), enPassant: true });
      }
    }
  } else if (type === 'N') {
    for (const [dr, dc] of [[-2,-1],[-2,1],[-1,-2],[-1,2],[1,-2],[1,2],[2,-1],[2,1]]) {
      addMove(r + dr, c + dc);
    }
  } else if (type === 'B') {
    for (const [dr, dc] of [[-1,-1],[-1,1],[1,-1],[1,1]]) slide(dr, dc);
  } else if (type === 'R') {
    for (const [dr, dc] of [[-1,0],[1,0],[0,-1],[0,1]]) slide(dr, dc);
  } else if (type === 'Q') {
    for (const [dr, dc] of [[-1,-1],[-1,1],[1,-1],[1,1],[-1,0],[1,0],[0,-1],[0,1]]) slide(dr, dc);
  } else if (type === 'K') {
    for (const [dr, dc] of [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]]) addMove(r + dr, c + dc);
    // Castling
    const backRank = side === 'w' ? 7 : 0;
    if (r === backRank && c === 4) {
      const cr = state.castleRights;
      // Kingside
      if ((side === 'w' ? cr.wK : cr.bK) && !board[backRank][5] && !board[backRank][6]) {
        moves.push({ from: [r, c], to: [backRank, 6], piece, castling: 'K' });
      }
      // Queenside
      if ((side === 'w' ? cr.wQ : cr.bQ) && !board[backRank][3] && !board[backRank][2] && !board[backRank][1]) {
        moves.push({ from: [r, c], to: [backRank, 2], piece, castling: 'Q' });
      }
    }
  }

  return moves;
}

// ─── Apply move ───────────────────────────────────────────────────────────────

export function applyMove(state: GameState, move: Move): GameState {
  const next = cloneState(state);
  const { board } = next;
  const { from, to, piece, promotion, castling, enPassant: isEP } = move;
  const side = pieceColor(piece)!;
  const backRank = side === 'w' ? 7 : 0;

  // Move piece
  board[to[0]][to[1]] = promotion ? makePiece(side, promotion) : piece;
  board[from[0]][from[1]] = null;

  // En passant capture
  if (isEP) {
    const captureRow = side === 'w' ? to[0] + 1 : to[0] - 1;
    board[captureRow][to[1]] = null;
  }

  // Castling rook move
  if (castling === 'K') {
    board[backRank][5] = makePiece(side, 'R');
    board[backRank][7] = null;
  } else if (castling === 'Q') {
    board[backRank][3] = makePiece(side, 'R');
    board[backRank][0] = null;
  }

  // Update castling rights
  if (pieceType(piece) === 'K') {
    if (side === 'w') { next.castleRights.wK = false; next.castleRights.wQ = false; }
    else { next.castleRights.bK = false; next.castleRights.bQ = false; }
  }
  if (pieceType(piece) === 'R') {
    if (from[0] === 7 && from[1] === 7) next.castleRights.wK = false;
    if (from[0] === 7 && from[1] === 0) next.castleRights.wQ = false;
    if (from[0] === 0 && from[1] === 7) next.castleRights.bK = false;
    if (from[0] === 0 && from[1] === 0) next.castleRights.bQ = false;
  }

  // En passant target
  next.enPassant = null;
  if (pieceType(piece) === 'P' && Math.abs(to[0] - from[0]) === 2) {
    next.enPassant = [(from[0] + to[0]) / 2, from[1]];
  }

  // Clocks
  if (pieceType(piece) === 'P' || move.captured) next.halfMoveClock = 0;
  else next.halfMoveClock++;
  if (side === 'b') next.fullMoveNumber++;

  next.turn = opponent(side);
  return next;
}

// ─── Check detection ─────────────────────────────────────────────────────────

export function findKing(board: (Piece | null)[][], side: Color): Square | null {
  const king = makePiece(side, 'K');
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) if (board[r][c] === king) return [r, c];
  return null;
}

export function isSquareAttacked(state: GameState, sq: Square, byColor: Color): boolean {
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (pieceColor(state.board[r][c]) !== byColor) continue;
      const raw = getRawMoves({ ...state, turn: byColor }, r, c);
      if (raw.some(m => m.to[0] === sq[0] && m.to[1] === sq[1] && !m.castling)) return true;
    }
  }
  return false;
}

export function isInCheck(state: GameState, side: Color): boolean {
  const kingSq = findKing(state.board, side);
  if (!kingSq) return false;
  return isSquareAttacked(state, kingSq, opponent(side));
}

// ─── Legal moves ─────────────────────────────────────────────────────────────

export function getLegalMoves(state: GameState, r: number, c: number): Move[] {
  const piece = state.board[r][c];
  if (!piece) return [];
  const side = pieceColor(piece)!;

  return getRawMoves(state, r, c).filter(move => {
    // Castling: king must not pass through check
    if (move.castling) {
      const passCol = move.castling === 'K' ? 5 : 3;
      const backRank = side === 'w' ? 7 : 0;
      if (isInCheck(state, side)) return false;
      if (isSquareAttacked(state, [backRank, passCol], opponent(side))) return false;
    }
    const next = applyMove(state, move);
    return !isInCheck(next, side);
  });
}

export function getAllLegalMoves(state: GameState): Move[] {
  const moves: Move[] = [];
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      if (pieceColor(state.board[r][c]) === state.turn) {
        moves.push(...getLegalMoves(state, r, c));
      }
    }
  }
  return moves;
}

export function getGameResult(state: GameState): 'ongoing' | 'checkmate' | 'stalemate' | 'draw-50' {
  if (state.halfMoveClock >= 100) return 'draw-50';
  const moves = getAllLegalMoves(state);
  if (moves.length === 0) {
    return isInCheck(state, state.turn) ? 'checkmate' : 'stalemate';
  }
  return 'ongoing';
}

// ─── Algebraic notation ───────────────────────────────────────────────────────

const FILES = 'abcdefgh';
export function squareToAlg(sq: Square): string {
  return `${FILES[sq[1]]}${8 - sq[0]}`;
}
export function moveToAlg(move: Move): string {
  const from = squareToAlg(move.from);
  const to = squareToAlg(move.to);
  const prom = move.promotion ? move.promotion.toLowerCase() : '';
  return `${from}${to}${prom}`;
}
