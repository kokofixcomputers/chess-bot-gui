/**
 * Strong Chess AI - Ready in 30 Minutes
 * Features:
 * - Pre-tuned evaluation (PeSTO-style)
 * - Aggressive move ordering
 * - Late Move Reductions
 * - Null move pruning
 * - Razoring
 * - Opening book of master games
 * - Simple endgame knowledge
 */

import type { GameState, Move } from './types';
import { applyMove, getAllLegalMoves, isInCheck, pieceType, pieceColor, getGameResult } from './chess';

// ─── Piece values (more accurate) ────────────────────────────────────────────

const PIECE_VALUE: Record<string, number> = {
  P: 100, N: 320, B: 330, R: 500, Q: 900, K: 20000,  // King value for safety
};

// ─── Pre-computed Piece-Square Tables (PeSTO - proven strong) ───────────────

// Midgame tables (from PeSTO evaluation)
const MG_PAWN = [
  0,   0,   0,   0,   0,   0,   0,   0,
  50, 50, 50, 50, 50, 50, 50, 50,
  10, 10, 20, 30, 30, 20, 10, 10,
  5,  5, 10, 27, 27, 10,  5,  5,
  0,  0,  0, 25, 25,  0,  0,  0,
  5, -5,-10,  0,  0,-10, -5,  5,
  5, 10, 10,-25,-25, 10, 10,  5,
  0,  0,  0,  0,  0,  0,  0,  0
];

const MG_KNIGHT = [
  -50,-40,-30,-30,-30,-30,-40,-50,
  -40,-20,  0,  0,  0,  0,-20,-40,
  -30,  0, 10, 15, 15, 10,  0,-30,
  -30,  5, 15, 20, 20, 15,  5,-30,
  -30,  0, 15, 20, 20, 15,  0,-30,
  -30,  5, 10, 15, 15, 10,  5,-30,
  -40,-20,  0,  5,  5,  0,-20,-40,
  -50,-40,-30,-30,-30,-30,-40,-50
];

const MG_BISHOP = [
  -20,-10,-10,-10,-10,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5, 10, 10,  5,  0,-10,
  -10,  5,  5, 10, 10,  5,  5,-10,
  -10,  0, 10, 10, 10, 10,  0,-10,
  -10, 10, 10, 10, 10, 10, 10,-10,
  -10,  5,  0,  0,  0,  0,  5,-10,
  -20,-10,-10,-10,-10,-10,-10,-20
];

const MG_ROOK = [
   0,  0,  0,  0,  0,  0,  0,  0,
   5, 10, 10, 10, 10, 10, 10,  5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
  -5,  0,  0,  0,  0,  0,  0, -5,
   0,  0,  0,  5,  5,  0,  0,  0
];

const MG_QUEEN = [
  -20,-10,-10, -5, -5,-10,-10,-20,
  -10,  0,  0,  0,  0,  0,  0,-10,
  -10,  0,  5,  5,  5,  5,  0,-10,
   -5,  0,  5,  5,  5,  5,  0, -5,
    0,  0,  5,  5,  5,  5,  0, -5,
  -10,  5,  5,  5,  5,  5,  0,-10,
  -10,  0,  5,  0,  0,  0,  0,-10,
  -20,-10,-10, -5, -5,-10,-10,-20
];

const MG_KING = [
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -30,-40,-40,-50,-50,-40,-40,-30,
  -20,-30,-30,-40,-40,-30,-30,-20,
  -10,-20,-20,-20,-20,-20,-20,-10,
   20, 20,  0,  0,  0,  0, 20, 20,
   20, 30, 10,  0,  0, 10, 30, 20
];

// Endgame tables (king becomes more active)
const EG_KING = [
  -50,-40,-30,-20,-20,-30,-40,-50,
  -30,-20,-10,  0,  0,-10,-20,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 30, 40, 40, 30,-10,-30,
  -30,-10, 20, 30, 30, 20,-10,-30,
  -30,-30,  0,  0,  0,  0,-30,-30,
  -50,-30,-30,-30,-30,-30,-30,-50
];

// Combined PST map
const PST: Record<string, number[]> = {
  P: MG_PAWN, N: MG_KNIGHT, B: MG_BISHOP, 
  R: MG_ROOK, Q: MG_QUEEN, K: MG_KING
};

// ─── Game phase detection ────────────────────────────────────────────────────

function getGamePhase(state: GameState): number {
  let pieces = 0;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (p && p[1] !== 'P' && p[1] !== 'K') {
        pieces++;
      }
    }
  }
  // 0 = endgame, 1 = middlegame
  return Math.min(1, pieces / 20);
}

// ─── Enhanced evaluation ─────────────────────────────────────────────────────

export function evaluate(state: GameState): number {
  const result = getGameResult(state);
  if (result === 'checkmate') return state.turn === 'w' ? -99999 : 99999;
  if (result === 'stalemate' || result === 'draw-50') return 0;

  let score = 0;
  const phase = getGamePhase(state);
  
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (!p) continue;
      
      const color = pieceColor(p);
      const pt = pieceType(p);
      const sq = r * 8 + c;
      const mirrorSq = color === 'b' ? 63 - sq : sq;
      
      // Material
      let val = PIECE_VALUE[pt];
      
      // Position (interpolate between midgame and endgame)
      if (pt === 'K') {
        // King uses different tables for mid/endgame
        const mgPos = MG_KING[mirrorSq];
        const egPos = EG_KING[mirrorSq];
        val += mgPos * (1 - phase) + egPos * phase;
      } else {
        val += PST[pt][mirrorSq];
      }
      
      // Bonus for castling (encourage king safety)
      if (pt === 'K') {
        if ((color === 'w' && r === 0 && (c === 6 || c === 2)) ||
            (color === 'b' && r === 7 && (c === 6 || c === 2))) {
          val += 50; // Castled king bonus
        }
      }
      
      // Bonus for developed pieces (first rank)
      if (pt !== 'P' && pt !== 'K') {
        if ((color === 'w' && r > 0) || (color === 'b' && r < 7)) {
          val += 10; // Developed piece bonus
        }
      }
      
      // Penalty for blocked pawns
      if (pt === 'P') {
        const forward = color === 'w' ? 1 : -1;
        if (r + forward >= 0 && r + forward < 8) {
          const ahead = state.board[r + forward][c];
          if (ahead) val -= 20; // Blocked pawn
        }
      }
      
      score += color === 'w' ? val : -val;
    }
  }

  // Bonus for center control (simplified)
  const centerSquares = [[3,3], [3,4], [4,3], [4,4]];
  for (const [r, c] of centerSquares) {
    const p = state.board[r][c];
    if (p) {
      score += pieceColor(p) === 'w' ? 15 : -15;
    }
  }

  return score;
}

// ─── Opening Book (most common master lines) ─────────────────────────────────

const OPENING_BOOK: Record<string, string[]> = {
  // e4 openings
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],
  // e4 e5
  'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2': ['g1f3', 'f1c4', 'd2d4'],
  // Italian
  'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3': ['f1c4'],
  // Ruy Lopez
  'r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3': ['a7a6', 'g8f6'],
  
  // d4 openings
  'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1': ['g8f6', 'd7d5', 'e7e6'],
  // d4 Nf6
  'rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2': ['c2c4', 'g1f3'],
  // Queen's Gambit
  'rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2': ['e7e6', 'c7c6'],
};

// ─── Move ordering scores ────────────────────────────────────────────────────

function scoreMove(state: GameState, move: Move): number {
  let score = 0;
  
  // 1. PV move (from previous iteration) - highest priority
  // (handled separately)
  
  // 2. Captures (MVV-LVA)
  if (move.captured) {
    const victim = PIECE_VALUE[move.captured[1]] || 0;
    const attacker = PIECE_VALUE[move.piece[1]] || 0;
    score = 10000 + victim * 10 - attacker;
  }
  
  // 3. Promotions
  else if (move.promotion) {
    score = 8000 + PIECE_VALUE[move.promotion] || 0;
  }
  
  // 4. Checks
  else if (isInCheck(applyMove(state, move))) {
    score = 5000;
  }
  
  // 5. Center control
  else {
    const toRank = move.to.r;
    const toFile = move.to.c;
    if (toRank >= 3 && toRank <= 4 && toFile >= 3 && toFile <= 4) {
      score = 1000;
    }
    
    // 6. Development
    else if (move.piece[1] !== 'P' && move.piece[1] !== 'K') {
      if ((move.piece[0] === 'w' && move.from.r === 0) ||
          (move.piece[0] === 'b' && move.from.r === 7)) {
        score = 500;
      }
    }
  }
  
  return score;
}

// ─── Search with all optimizations ───────────────────────────────────────────

interface SearchInfo {
  nodes: number;
  bestMove: Move | null;
  pv: Move[];
}

class StrongSearcher {
  private nodes = 0;
  private startTime = 0;
  private timeLimit = 0;
  private pvTable: Move[][] = [];
  private killerMoves: Move[][] = []; // [ply][slot]
  private historyMoves: number[][][] = []; // [piece][from][to]
  
  constructor() {
    // Initialize killer moves (2 per ply)
    for (let i = 0; i < 64; i++) {
      this.killerMoves[i] = [];
    }
    // Initialize history table
    for (let p = 0; p < 12; p++) {
      this.historyMoves[p] = [];
      for (let f = 0; f < 64; f++) {
        this.historyMoves[p][f] = new Array(64).fill(0);
      }
    }
  }
  
  public search(state: GameState, timeLimitMs: number): SearchInfo {
    this.startTime = Date.now();
    this.timeLimit = timeLimitMs;
    this.nodes = 0;
    this.pvTable = [];
    
    const moves = getAllLegalMoves(state);
    if (moves.length === 0) return { nodes: 0, bestMove: null, pv: [] };
    
    let bestMove = moves[0];
    let bestScore = -Infinity;
    let pv: Move[] = [];
    
    // Iterative deepening
    for (let depth = 1; depth <= 20; depth++) {
      // Check time
      if (Date.now() - this.startTime > this.timeLimit) break;
      
      // Clear PV for new depth
      this.pvTable[depth] = [];
      
      // Aspiration windows
      let alpha = -Infinity;
      let beta = Infinity;
      let delta = 50;
      
      for (let attempt = 0; attempt < 3; attempt++) {
        if (attempt > 0) {
          alpha = Math.max(-Infinity, bestScore - delta);
          beta = Math.min(Infinity, bestScore + delta);
          delta *= 2;
        }
        
        const score = this.negamax(state, depth, alpha, beta, 0);
        
        if (score <= alpha) {
          // Failed low - research with wider window
          continue;
        }
        if (score >= beta) {
          // Failed high - research with wider window
          continue;
        }
        
        bestScore = score;
        bestMove = this.pvTable[0]?.[0] || bestMove;
        pv = [...(this.pvTable[0] || [])];
        break;
      }
    }
    
    return {
      nodes: this.nodes,
      bestMove,
      pv
    };
  }
  
  private negamax(
    state: GameState, 
    depth: number, 
    alpha: number, 
    beta: number,
    ply: number
  ): number {
    this.nodes++;
    
    // Time check (every 1000 nodes)
    if (this.nodes % 1000 === 0) {
      if (Date.now() - this.startTime > this.timeLimit) {
        return 0; // Timeout
      }
    }
    
    // Check for game end
    const result = getGameResult(state);
    if (result !== 'ongoing') {
      if (result === 'checkmate') return -99999 + ply; // Prefer faster mates
      return 0;
    }
    
    // Quiescence search at leaf
    if (depth <= 0) {
      return this.quiescence(state, alpha, beta, ply);
    }
    
    // Null move pruning (skip if not in check and not zugzwang)
    if (depth >= 3 && !isInCheck(state) && this.hasNonPawnMaterial(state)) {
      const R = 2 + Math.min(3, depth / 4);
      const nullState = { ...state, turn: state.turn === 'w' ? 'b' : 'w' };
      const score = -this.negamax(nullState, depth - 1 - R, -beta, -beta + 1, ply + 1);
      if (score >= beta) return beta;
    }
    
    // Razoring (if position is very bad, prune)
    if (depth <= 3 && !isInCheck(state)) {
      const evalScore = evaluate(state);
      const margin = 300 * depth;
      if (evalScore + margin < alpha) {
        const qScore = this.quiescence(state, alpha - margin, beta, ply);
        if (qScore <= alpha) return qScore;
      }
    }
    
    // Get moves with ordering
    const moves = this.orderMoves(state, ply);
    if (moves.length === 0) {
      return isInCheck(state) ? -99999 + ply : 0;
    }
    
    let bestScore = -Infinity;
    let movesSearched = 0;
    
    for (const move of moves) {
      const next = applyMove(state, move);
      
      // Late Move Reduction
      let reduction = 0;
      if (depth >= 3 && movesSearched >= 4 && !move.captured && !isInCheck(state)) {
        reduction = 1;
        if (movesSearched >= 8) reduction = 2;
      }
      
      let score: number;
      if (reduction > 0) {
        // Search with reduced depth
        score = -this.negamax(next, depth - 1 - reduction, -alpha - 1, -alpha, ply + 1);
        
        // If score beats alpha, research with full depth
        if (score > alpha) {
          score = -this.negamax(next, depth - 1, -alpha - 1, -alpha, ply + 1);
        }
      } else {
        score = -this.negamax(next, depth - 1, -alpha - 1, -alpha, ply + 1);
      }
      
      // PV Search - full window for good moves
      if (score > alpha && score < beta && movesSearched > 0) {
        score = -this.negamax(next, depth - 1, -beta, -alpha, ply + 1);
      }
      
      if (score > bestScore) {
        bestScore = score;
        
        // Update PV
        this.pvTable[ply] = [move, ...(this.pvTable[ply + 1] || [])];
        
        // Update killer moves (for non-captures)
        if (!move.captured) {
          this.killerMoves[ply][1] = this.killerMoves[ply][0];
          this.killerMoves[ply][0] = move;
        }
      }
      
      if (score > alpha) {
        alpha = score;
        
        // Update history (for non-captures)
        if (!move.captured) {
          const pieceIdx = this.getPieceIndex(move.piece);
          const fromSq = move.from.r * 8 + move.from.c;
          const toSq = move.to.r * 8 + move.to.c;
          this.historyMoves[pieceIdx][fromSq][toSq] += depth * depth;
        }
      }
      
      if (alpha >= beta) {
        break; // Beta cutoff
      }
      
      movesSearched++;
    }
    
    return bestScore;
  }
  
  private quiescence(state: GameState, alpha: number, beta: number, ply: number): number {
    this.nodes++;
    
    // Stand-pat evaluation
    const standPat = evaluate(state);
    if (standPat >= beta) return beta;
    if (standPat > alpha) alpha = standPat;
    
    // Only consider captures and promotions
    const moves = getAllLegalMoves(state)
      .filter(m => m.captured || m.promotion || isInCheck(applyMove(state, m)))
      .sort((a, b) => {
        const aScore = a.captured ? (PIECE_VALUE[a.captured[1]] || 0) : 0;
        const bScore = b.captured ? (PIECE_VALUE[b.captured[1]] || 0) : 0;
        return bScore - aScore;
      });
    
    for (const move of moves) {
      const next = applyMove(state, move);
      const score = -this.quiescence(next, -beta, -alpha, ply + 1);
      
      if (score >= beta) return beta;
      if (score > alpha) alpha = score;
    }
    
    return alpha;
  }
  
  private orderMoves(state: GameState, ply: number): Move[] {
    const moves = getAllLegalMoves(state);
    const ttMove = this.pvTable[ply]?.[0]; // PV move from previous iteration
    
    return moves.sort((a, b) => {
      // PV move first
      if (ttMove && this.movesEqual(a, ttMove)) return Infinity;
      if (ttMove && this.movesEqual(b, ttMove)) return -Infinity;
      
      const aScore = this.scoreMoveForOrdering(state, a, ply);
      const bScore = this.scoreMoveForOrdering(state, b, ply);
      return bScore - aScore;
    });
  }
  
  private scoreMoveForOrdering(state: GameState, move: Move, ply: number): number {
    // Captures (MVV-LVA)
    if (move.captured) {
      const victim = PIECE_VALUE[move.captured[1]] || 0;
      const attacker = PIECE_VALUE[move.piece[1]] || 0;
      return 10000 + victim * 10 - attacker;
    }
    
    // Promotions
    if (move.promotion) {
      return 8000 + PIECE_VALUE[move.promotion];
    }
    
    // Killer moves
    if (this.killerMoves[ply]?.some(k => k && this.movesEqual(move, k))) {
      return 5000;
    }
    
    // History heuristic
    const pieceIdx = this.getPieceIndex(move.piece);
    const fromSq = move.from.r * 8 + move.from.c;
    const toSq = move.to.r * 8 + move.to.c;
    const history = this.historyMoves[pieceIdx]?.[fromSq]?.[toSq] || 0;
    if (history > 0) {
      return 1000 + history;
    }
    
    // Positional (center control, development)
    const toRank = move.to.r;
    const toFile = move.to.c;
    if (toRank >= 3 && toRank <= 4 && toFile >= 3 && toFile <= 4) {
      return 500;
    }
    
    return 0;
  }
  
  private movesEqual(a: Move, b: Move): boolean {
    return a.from.r === b.from.r && a.from.c === b.from.c &&
           a.to.r === b.to.r && a.to.c === b.to.c &&
           a.promotion === b.promotion;
  }
  
  private getPieceIndex(piece: string): number {
    const pieces = ['P', 'N', 'B', 'R', 'Q', 'K'];
    const color = piece[0];
    const pt = piece[1];
    const base = pieces.indexOf(pt);
    return color === 'w' ? base : base + 6;
  }
  
  private hasNonPawnMaterial(state: GameState): boolean {
    for (let r = 0; r < 8; r++) {
      for (let c = 0; c < 8; c++) {
        const p = state.board[r][c];
        if (p && p[1] !== 'P' && p[1] !== 'K') return true;
      }
    }
    return false;
  }
}

// ─── Public API ───────────────────────────────────────────────────────────────

export interface AIResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
  pv: Move[];
}

export async function getBestMove(state: GameState, timeLimitMs = 2000): Promise<AIResult> {
  return new Promise(resolve => {
    // Small delay to not block UI
    setTimeout(() => {
      // Check opening book first
      const fen = boardToFen(state);
      if (OPENING_BOOK[fen]) {
        const bookMoves = OPENING_BOOK[fen];
        const moves = getAllLegalMoves(state);
        
        // Find matching book move
        for (const bookMove of bookMoves) {
          const move = moves.find(m => 
            algebraicToMove(m, bookMove, state)
          );
          if (move) {
            resolve({
              move,
              score: 0,
              depth: 0,
              nodes: 0,
              pv: [move]
            });
            return;
          }
        }
      }
      
      // Search for best move
      const searcher = new StrongSearcher();
      const result = searcher.search(state, timeLimitMs);
      
      if (!result.bestMove) {
        // Fallback to first legal move
        const moves = getAllLegalMoves(state);
        resolve({
          move: moves[0],
          score: 0,
          depth: 0,
          nodes: 0,
          pv: []
        });
        return;
      }
      
      resolve({
        move: result.bestMove,
        score: 0,
        depth: 0,
        nodes: result.nodes,
        pv: result.pv
      });
    }, 10);
  });
}

// Helper to convert board to FEN (simplified)
function boardToFen(state: GameState): string {
  let fen = '';
  for (let r = 0; r < 8; r++) {
    let empty = 0;
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (p) {
        if (empty > 0) {
          fen += empty;
          empty = 0;
        }
        fen += p[1] === p[1].toUpperCase() ? p[1] : p[1].toLowerCase();
      } else {
        empty++;
      }
    }
    if (empty > 0) fen += empty;
    if (r < 7) fen += '/';
  }
  fen += ` ${state.turn} - - 0 1`; // Simplified
  return fen;
}

// Helper to convert algebraic notation to Move object
function algebraicToMove(move: Move, alg: string, state: GameState): boolean {
  // Simplified - just check if the move matches the algebraic notation
  // In production, you'd want proper conversion
  const fromFile = String.fromCharCode(97 + move.from.c);
  const fromRank = move.from.r + 1;
  const toFile = String.fromCharCode(97 + move.to.c);
  const toRank = move.to.r + 1;
  const moveStr = `${fromFile}${fromRank}${toFile}${toRank}`;
  return moveStr === alg;
}
