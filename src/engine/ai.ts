/**
 * Chess AI Engine
 * ---------------
 * Implements a classical minimax search with:
 *  - Alpha-beta pruning (negamax formulation)
 *  - Iterative deepening  
 *  - Transposition table (Zobrist hash based)
 *  - Move ordering (TT hit, check, MVV-LVA, promotion)
 *  - Piece-square tables for positional evaluation
 *  - Quiescence search to avoid horizon effect
 */

import type { GameState, Move } from './types';
import { applyMove, getAllLegalMoves, isInCheck, pieceType, pieceColor, getGameResult } from './chess';

// ─── Piece values ─────────────────────────────────────────────────────────────

const PIECE_VALUE: Record<string, number> = {
  P: 100, N: 320, B: 330, R: 500, Q: 900, K: 0,
};

// ─── Piece-Square Tables (flattened, white's perspective, a1=0, h1=7, a8=56, h8=63) ─────────

const PAWN_PST = [0,0,0,0,0,0,0,0,50,50,50,50,50,50,50,50,10,10,20,30,30,20,10,10,5,5,10,25,25,10,5,5,0,0,0,20,20,0,0,0,5,-5,-10,0,0,-10,-5,5,5,10,10,-20,-20,10,10,5,0,0,0,0,0,0,0,0];
const KNIGHT_PST = [-50,-40,-30,-30,-30,-30,-40,-50,-40,-20,0,0,0,0,-20,-40,-30,0,10,15,15,10,0,-30,-30,5,15,20,20,15,5,-30,-30,0,15,20,20,15,0,-30,-30,5,10,15,15,10,5,-30,-40,-20,0,5,5,0,-20,-40,-50,-40,-30,-30,-30,-30,-40,-50];
const BISHOP_PST = [-20,-10,-10,-10,-10,-10,-10,-20,-10,5,0,0,0,0,5,-10,-10,10,10,10,10,10,10,-10,-10,0,10,10,10,10,0,-10,-10,5,5,10,10,5,5,-10,-10,0,5,10,10,5,0,-10,-10,0,0,0,0,0,0,-10,-20,-10,-10,-10,-10,-10,-10,-20];
const ROOK_PST = [0,0,0,5,5,0,0,0,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,-5,0,0,0,0,0,0,-5,5,10,10,10,10,10,10,5,0,0,0,0,0,0,0,0];
const QUEEN_PST = [-20,-10,-10,-5,-5,-10,-10,-20,-10,0,5,0,0,0,0,-10,-10,5,5,5,5,5,0,-10,-5,0,5,5,5,5,0,-5,0,0,5,5,5,5,0,-5,-10,5,5,5,5,5,0,-10,-10,0,5,0,0,0,0,-10,-20,-10,-10,-5,-5,-10,-10,-20];
const KING_PST = [-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-30,-40,-40,-50,-50,-40,-40,-30,-20,-30,-30,-40,-40,-30,-30,-20,-10,-20,-20,-20,-20,-20,-20,-10,20,20,0,0,0,0,20,20,20,30,10,0,0,10,30,20];

const PST_BY_PIECE: Record<string, number[]> = {
  P: PAWN_PST, N: KNIGHT_PST, B: BISHOP_PST, R: ROOK_PST, Q: QUEEN_PST, K: KING_PST
};

// ─── Zobrist Hashing ─────────────────────────────────────────────────────────

const ZOBRIST_PIECES = new Uint32Array(12 * 64);
const ZOBRIST_BLACK = 1n << 32n;
const ZOBRIST_SIDE = BigInt('0x123456789ABCDEF0');

function initZobrist() {
  for (let i = 0; i < ZOBRIST_PIECES.length; i++) {
    ZOBRIST_PIECES[i] = Math.floor(Math.random() * 0x100000000);
  }
}
initZobrist();

function computeZobrist(state: GameState): bigint {
  let hash = 0n;
  
  // Board pieces
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (p) {
        const pt = pieceType(p);
        const pc = pieceColor(p);
        const pieceIdx = (pc === 'w' ? 0 : 6) + 'PNBRQK'.indexOf(pt);
        const sq = r * 8 + c;
        hash ^= BigInt(ZOBRIST_PIECES[pieceIdx * 64 + sq]);
      }
    }
  }
  
  // Side to move
  if (state.turn === 'b') hash ^= ZOBRIST_SIDE;
  
  return hash;
}

// ─── Transposition Table ─────────────────────────────────────────────────────

interface TTEntry {
  depth: number;
  score: number;
  flag: 0 | 1 | 2; // 0=exact, 1=lowerbound, 2=upperbound
  move?: Move;
}

const TT_SIZE = 1 << 20; // 1M entries
const TT = new Map<bigint, TTEntry>();

function getTTMove(hash: bigint): Move | undefined {
  const entry = TT.get(hash);
  return entry?.move;
}

function storeTT(hash: bigint, depth: number, score: number, flag: 0 | 1 | 2, move?: Move) {
  if (!TT.has(hash) || depth >= (TT.get(hash)?.depth ?? 0)) {
    TT.set(hash, { depth, score, flag, move });
    if (TT.size > TT_SIZE) {
      // Simple eviction: remove oldest
      const firstKey = TT.keys().next().value;
      TT.delete(firstKey);
    }
  }
}

// ─── Evaluation ───────────────────────────────────────────────────────────────

export function evaluate(state: GameState): number {
  const result = getGameResult(state);
  if (result === 'checkmate') return state.turn === 'w' ? -99999 : 99999;
  if (result === 'stalemate' || result === 'draw-50') return 0;

  let score = 0;
  
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (!p) continue;
      
      const color = pieceColor(p);
      const pt = pieceType(p);
      const sq = r * 8 + c;
      
      const pst = color === 'b' ? PST_BY_PIECE[pt][63 - sq] : PST_BY_PIECE[pt][sq];
      const val = (PIECE_VALUE[pt] ?? 0) + pst;
      
      score += color === 'w' ? val : -val;
    }
  }

  return score;
}

// ─── Move ordering ────────────────────────────────────────────────────────────

function scoreMoveForOrdering(state: GameState, move: Move, ttMove?: Move): number {
  if (ttMove && areMovesEqual(move, ttMove)) return 1000000;
  
  if (isInCheck(applyMove(state, move))) return 50000;
  
  if (move.captured) {
    const attacker = PIECE_VALUE[move.piece[1]] ?? 0;
    const victim = PIECE_VALUE[move.captured[1]] ?? 0;
    return 10000 + (victim * 10 - attacker);
  }
  
  if (move.promotion) return 20000 + PIECE_VALUE[move.promotion ?? 'Q'];
  
  return 0;
}

function areMovesEqual(a: Move, b: Move): boolean {
  return a.from.r === b.from.r && a.from.c === b.from.c &&
         a.to.r === b.to.r && a.to.c === b.to.c &&
         a.promotion === b.promotion;
}

function orderMoves(state: GameState, moves: Move[], ttMove?: Move): Move[] {
  return [...moves].sort((a, b) => 
    scoreMoveForOrdering(state, b, ttMove) - scoreMoveForOrdering(state, a, ttMove)
  );
}

// ─── Quiescence search ────────────────────────────────────────────────────────

function quiescence(state: GameState, alpha: number, beta: number): number {
  const stand = evaluate(state);
  if (stand >= beta) return beta;
  if (stand > alpha) alpha = stand;

  const forcingMoves = getAllLegalMoves(state).filter(m => 
    m.captured || isInCheck(applyMove(state, m))
  );
  
  for (const move of orderMoves(state, forcingMoves)) {
    const next = applyMove(state, move);
    const score = -quiescence(next, -beta, -alpha);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }
  return alpha;
}

// ─── Negamax with alpha-beta pruning ─────────────────────────────────────────

let nodesSearched = 0;

function negamax(state: GameState, depth: number, alpha: number, beta: number): number {
  nodesSearched++;
  
  const hash = computeZobrist(state);
  const ttEntry = TT.get(hash);
  
  // TT lookup
  if (ttEntry && ttEntry.depth >= depth) {
    if (ttEntry.flag === 0) return ttEntry.score; // exact
    if (ttEntry.flag === 1 && ttEntry.score >= beta) return ttEntry.score; // lowerbound
    if (ttEntry.flag === 2 && ttEntry.score <= alpha) return ttEntry.score; // upperbound
  }
  
  const ttMove = ttEntry?.move;
  
  const result = getGameResult(state);
  if (result !== 'ongoing') {
    if (result === 'checkmate') return -99999;
    return 0;
  }
  
  if (depth === 0) return quiescence(state, alpha, beta);
  
  let bestScore = -Infinity;
  
  const moves = orderMoves(state, getAllLegalMoves(state), ttMove);
  for (const move of moves) {
    const next = applyMove(state, move);
    const score = -negamax(next, depth - 1, -beta, -alpha);
    
    if (score > bestScore) bestScore = score;
    if (bestScore > alpha) alpha = bestScore;
    if (alpha >= beta) break;
  }
  
  // Store in TT
  let flag = 0; // exact
  if (bestScore <= alpha) flag = 2; // upperbound
  else if (bestScore >= beta) flag = 1; // lowerbound
  
  storeTT(hash, depth, bestScore, flag, moves[0]); // store first (best) move
  
  return bestScore;
}

// ─── Public API ───────────────────────────────────────────────────────────────

export interface AIResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

export async function getBestMove(state: GameState, timeLimitMs = 1500): Promise<AIResult> {
  return new Promise(resolve => {
    setTimeout(() => {
      const start = Date.now();
      nodesSearched = 0;
      
      const hash = computeZobrist(state);
      TT.clear(); // Clear TT for new search
      
      const moves = getAllLegalMoves(state);
      if (moves.length === 0) throw new Error('No legal moves');
      if (moves.length === 1) {
        resolve({ move: moves[0], score: 0, depth: 0, nodes: 1 });
        return;
      }
      
      let bestMove = moves[0];
      let bestScore = -Infinity;
      let completedDepth = 0;
      
      // Iterative deepening
      for (let depth = 1; depth <= 8; depth++) {
        if (Date.now() - start > timeLimitMs) break;
        
        let depthBestMove = moves[0];
        let depthBestScore = -Infinity;
        let alpha = -Infinity;
        let beta = Infinity;
        
        const ttMove = getTTMove(hash);
        const ordered = orderMoves(state, moves, ttMove);
        
        for (const move of ordered) {
          if (Date.now() - start > timeLimitMs) break;
          
          const next = applyMove(state, move);
          const score = -negamax(next, depth - 1, -beta, -alpha);
          
          if (score > depthBestScore) {
            depthBestScore = score;
            depthBestMove = move;
          }
          
          if (score > alpha) alpha = score;
        }
        
        if (depthBestMove) {
          bestMove = depthBestMove;
          bestScore = depthBestScore;
          completedDepth = depth;
        }
      }
      
      resolve({ move: bestMove, score: bestScore, depth: completedDepth, nodes: nodesSearched });
    }, 10);
  });
}
