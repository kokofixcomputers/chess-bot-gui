/**
 * Chess AI Engine (fixed)
 * -----------------------
 * - Negamax + alpha-beta pruning
 * - Iterative deepening
 * - Zobrist hashing + transposition table
 * - NNUE evaluation using all available input features (768 or 784)
 */

import type { GameState, Move } from './types';
import {
  applyMove,
  getAllLegalMoves,
  isInCheck,
  pieceType,
  pieceColor,
  getGameResult,
} from './chess';

// ─── Piece values (for move ordering) ─────────────────────────────────────────

const PIECE_VALUE: Record<string, number> = {
  P: 100,
  N: 320,
  B: 330,
  R: 500,
  Q: 900,
  K: 0,
};

// ─── NNUE Evaluator ───────────────────────────────────────────────────────────

interface NNUEWeights {
  fc1_weight: Float32Array[]; // [256][inputSize]
  fc1_bias: Float32Array;     // [256]
  fc2_weight: Float32Array;   // [256]
  fc2_bias: number;
}

class NNUEEvaluator {
  private weights: NNUEWeights | null = null;
  private accumulator: Float32Array = new Float32Array(256);
  private inputSize = 768; // 12 * 64; will be updated from file

  private pieceMap: Record<string, number> = {
    wP: 0,
    wN: 1,
    wB: 2,
    wR: 3,
    wQ: 4,
    wK: 5,
    bP: 6,
    bN: 7,
    bB: 8,
    bR: 9,
    bQ: 10,
    bK: 11,
  };

  public async loadWeights(url: string): Promise<void> {
    const resp = await fetch(url);
    const buffer = await resp.arrayBuffer();
    const data = new DataView(buffer);
    let offset = 0;

    // Magic "NNUE"
    const magic =
      String.fromCharCode(data.getUint8(0)) +
      String.fromCharCode(data.getUint8(1)) +
      String.fromCharCode(data.getUint8(2)) +
      String.fromCharCode(data.getUint8(3));
    if (magic !== 'NNUE') {
      throw new Error('Invalid NNUE file header');
    }
    offset += 4;

    const version = data.getInt32(offset, true);
    offset += 4;
    console.log(`Loading NNUE v${version}`);

    const readInt32 = () => {
      const v = data.getInt32(offset, true);
      offset += 4;
      return v;
    };
    const readFloat32Array = (len: number) => {
      const arr = new Float32Array(buffer, offset, len);
      offset += len * 4;
      return arr;
    };

    // fc1_weight: flattened [256 * inputSize]
    const fc1_len = readInt32();
    const inferredInputSize = fc1_len / 256;
    if (!Number.isInteger(inferredInputSize)) {
      throw new Error(`fc1_len ${fc1_len} not divisible by 256`);
    }
    this.inputSize = inferredInputSize;
    const fc1_weight: Float32Array[] = [];
    for (let i = 0; i < 256; i++) {
      const row = new Float32Array(
        buffer,
        offset + i * this.inputSize * 4,
        this.inputSize,
      );
      fc1_weight.push(row);
    }
    offset += fc1_len * 4;

    // fc1_bias [256]
    const fc1_bias_len = readInt32();
    if (fc1_bias_len !== 256) {
      console.warn(`Unexpected fc1_bias_len=${fc1_bias_len}, expected 256`);
    }
    const fc1_bias = readFloat32Array(fc1_bias_len);

    // fc2_weight [256]
    const fc2_weight_len = readInt32();
    if (fc2_weight_len !== 256) {
      console.warn(`Unexpected fc2_weight_len=${fc2_weight_len}, expected 256`);
    }
    const fc2_weight = readFloat32Array(fc2_weight_len);

    // fc2_bias [1]
    const fc2_bias = data.getFloat32(offset, true);
    offset += 4;

    this.weights = { fc1_weight, fc1_bias, fc2_weight, fc2_bias };
    console.log(`NNUE weights loaded, inputSize=${this.inputSize}`);
  }

  /**
   * Build feature vector from board and ply, then run the first NNUE layer.
   */
  public setPosition(state: GameState, ply: number): void {
    if (!this.weights) {
      this.accumulator.fill(0);
      return;
    }

    const features = new Float32Array(this.inputSize);

    // 1) Piece features: 12 * 64 = 768 (if inputSize >= 768)
    const pieceFeatureCount = Math.min(768, this.inputSize);
    if (pieceFeatureCount === 768) {
      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const p = state.board[r][c];
          if (!p) continue;
          const col = pieceColor(p);
          const pt = pieceType(p);
          const key = `${col}${pt}` as keyof typeof this.pieceMap;
          const base = this.pieceMap[key];
          if (base === undefined) continue;
          const sq = r * 8 + c;
          const idx = base * 64 + sq;
          if (idx >= 0 && idx < pieceFeatureCount) {
            features[idx] = 1.0;
          }
        }
      }
    }

    // 2) Extra features if network expects >768 inputs (e.g. 784)
    if (this.inputSize > 768) {
      const idxCheck = 768;
      const idxPly = 769;
      const idxTurn = 770;
      const idxWKingDist = 771;
      const idxBKingDist = 772;
      const idxMobility = 773;

      // Check flag
      if (idxCheck < this.inputSize) {
        features[idxCheck] = isInCheck(state) ? 1.0 : 0.0;
      }

      // Normalized ply (0..1)
      if (idxPly < this.inputSize) {
        features[idxPly] = Math.min(ply / 80, 1.0);
      }

      // Turn indicator
      if (idxTurn < this.inputSize) {
        features[idxTurn] = state.turn === 'w' ? 1.0 : 0.0;
      }

      // King distance to center
      const centerR = 3.5;
      const centerC = 3.5;
      let wKingR = -1,
        wKingC = -1,
        bKingR = -1,
        bKingC = -1;

      for (let r = 0; r < 8; r++) {
        for (let c = 0; c < 8; c++) {
          const p = state.board[r][c];
          if (!p) continue;
          if (pieceType(p) === 'K') {
            if (pieceColor(p) === 'w') {
              wKingR = r;
              wKingC = c;
            } else {
              bKingR = r;
              bKingC = c;
            }
          }
        }
      }

      if (idxWKingDist < this.inputSize && wKingR !== -1) {
        const dr = wKingR - centerR;
        const dc = wKingC - centerC;
        features[idxWKingDist] = Math.min(
          Math.sqrt(dr * dr + dc * dc) / 4.0,
          1.0,
        );
      }
      if (idxBKingDist < this.inputSize && bKingR !== -1) {
        const dr = bKingR - centerR;
        const dc = bKingC - centerC;
        features[idxBKingDist] = Math.min(
          Math.sqrt(dr * dr + dc * dc) / 4.0,
          1.0,
        );
      }

      // Mobility
      if (idxMobility < this.inputSize) {
        const moves = getAllLegalMoves(state);
        features[idxMobility] = Math.min(moves.length / 30.0, 1.0);
      }
    }

    // Run first NNUE layer: clipped ReLU
    const { fc1_weight, fc1_bias } = this.weights;
    for (let i = 0; i < 256; i++) {
      let sum = fc1_bias[i];
      const row = fc1_weight[i];
      for (let j = 0; j < this.inputSize; j++) {
        sum += row[j] * features[j];
      }
      if (sum < 0) sum = 0;
      else if (sum > 1) sum = 1;
      this.accumulator[i] = sum;
    }
  }

  public evaluate(): number {
    if (!this.weights) return 0;
    const { fc2_weight, fc2_bias } = this.weights;
    let out = fc2_bias;
    for (let i = 0; i < 256; i++) {
      out += this.accumulator[i] * fc2_weight[i];
    }
    // Assume output is in pawns from side-to-move perspective; scale to centipawns
    return out * 100;
  }
}

const evaluator = new NNUEEvaluator();

// ─── Zobrist Hashing + Transposition Table ────────────────────────────────────

const ZOBRIST_PIECES = new Uint32Array(12 * 64);
const ZOBRIST_SIDE = BigInt('0x123456789ABCDEF0');

function initZobrist() {
  for (let i = 0; i < ZOBRIST_PIECES.length; i++) {
    ZOBRIST_PIECES[i] = Math.floor(Math.random() * 0x100000000);
  }
}
initZobrist();

function computeZobrist(state: GameState): bigint {
  let hash = 0n;
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const p = state.board[r][c];
      if (!p) continue;
      const pt = pieceType(p);
      const pc = pieceColor(p);
      const pieceIdx = (pc === 'w' ? 0 : 6) + 'PNBRQK'.indexOf(pt);
      const sq = r * 8 + c;
      hash ^= BigInt(ZOBRIST_PIECES[pieceIdx * 64 + sq]);
    }
  }
  if (state.turn === 'b') hash ^= ZOBRIST_SIDE;
  return hash;
}

interface TTEntry {
  depth: number;
  score: number;
  flag: 0 | 1 | 2; // 0 exact, 1 lower bound, 2 upper bound
  move?: Move;
}

const TT_SIZE = 1 << 20;
const TT = new Map<bigint, TTEntry>();

function getTTMove(hash: bigint): Move | undefined {
  return TT.get(hash)?.move;
}

function storeTT(
  hash: bigint,
  depth: number,
  score: number,
  flag: 0 | 1 | 2,
  move?: Move,
) {
  const existing = TT.get(hash);
  if (!existing || depth >= existing.depth) {
    TT.set(hash, { depth, score, flag, move });
    if (TT.size > TT_SIZE) {
      const firstKey = TT.keys().next().value;
      TT.delete(firstKey);
    }
  }
}

// ─── Evaluation wrappers ──────────────────────────────────────────────────────

/**
 * Internal eval with ply, used by search.
 */
function evalWithPly(state: GameState, ply: number): number {
  const result = getGameResult(state);
  if (result === 'checkmate') {
    return state.turn === 'w' ? -99999 : 99999;
  }
  if (result === 'stalemate' || result === 'draw-50') {
    return 0;
  }

  evaluator.setPosition(state, ply);
  return evaluator.evaluate();
}

/**
 * Optional external eval if you want to inspect positions in the UI.
 * Uses ply = 0 here.
 */
export function evaluate(state: GameState): number {
  return evalWithPly(state, 0);
}

// ─── Move ordering ────────────────────────────────────────────────────────────

function areMovesEqual(a: Move | undefined, b: Move | undefined): boolean {
  if (!a || !b) return false;
  return (
    a.from.r === b.from.r &&
    a.from.c === b.from.c &&
    a.to.r === b.to.r &&
    a.to.c === b.to.c &&
    a.promotion === b.promotion
  );
}

function scoreMoveForOrdering(
  state: GameState,
  move: Move,
  ttMove?: Move,
): number {
  if (ttMove && areMovesEqual(move, ttMove)) return 1_000_000;

  let score = 0;

  const next = applyMove(state, move);
  if (isInCheck(next)) score += 50_000;

  if (move.captured) {
    const attacker = PIECE_VALUE[move.piece[1]] ?? 0;
    const victim = PIECE_VALUE[move.captured[1]] ?? 0;
    score += 10_000 + (victim * 10 - attacker);
  }

  if (move.promotion) {
    score += 20_000 + (PIECE_VALUE[move.promotion] ?? 0);
  }

  return score;
}

function orderMoves(state: GameState, moves: Move[], ttMove?: Move): Move[] {
  return [...moves].sort(
    (a, b) =>
      scoreMoveForOrdering(state, b, ttMove) -
      scoreMoveForOrdering(state, a, ttMove),
  );
}

// ─── Quiescence search ────────────────────────────────────────────────────────

let nodesSearched = 0;

function quiescence(
  state: GameState,
  alpha: number,
  beta: number,
  ply: number,
): number {
  const standPat = evalWithPly(state, ply);
  if (standPat >= beta) return beta;
  if (standPat > alpha) alpha = standPat;

  const forcingMoves = getAllLegalMoves(state).filter((m) => {
    const capture = !!m.captured;
    const givesCheck = isInCheck(applyMove(state, m));
    return capture || givesCheck;
  });

  if (forcingMoves.length === 0) return alpha;

  const ordered = orderMoves(state, forcingMoves);
  for (const move of ordered) {
    const next = applyMove(state, move);
    const score = -quiescence(next, -beta, -alpha, ply + 1);
    if (score >= beta) return beta;
    if (score > alpha) alpha = score;
  }

  return alpha;
}

// ─── Negamax + alpha-beta ─────────────────────────────────────────────────────

function negamax(
  state: GameState,
  depth: number,
  alpha: number,
  beta: number,
  ply: number,
): number {
  nodesSearched++;

  const hash = computeZobrist(state);
  const ttEntry = TT.get(hash);
  if (ttEntry && ttEntry.depth >= depth) {
    if (ttEntry.flag === 0) return ttEntry.score;
    if (ttEntry.flag === 1 && ttEntry.score >= beta) return ttEntry.score;
    if (ttEntry.flag === 2 && ttEntry.score <= alpha) return ttEntry.score;
  }
  const ttMove = ttEntry?.move;

  const result = getGameResult(state);
  if (result !== 'ongoing') {
    if (result === 'checkmate') return -99999;
    return 0;
  }

  if (depth === 0) {
    return quiescence(state, alpha, beta, ply);
  }

  let bestScore = -Infinity;
  let bestMove: Move | undefined;

  const moves = orderMoves(state, getAllLegalMoves(state), ttMove);
  for (const move of moves) {
    const next = applyMove(state, move);
    const score = -negamax(next, depth - 1, -beta, -alpha, ply + 1);

    if (score > bestScore) {
      bestScore = score;
      bestMove = move;
    }
    if (bestScore > alpha) alpha = bestScore;
    if (alpha >= beta) break;
  }

  let flag: 0 | 1 | 2 = 0;
  if (bestScore <= alpha) flag = 2;
  else if (bestScore >= beta) flag = 1;

  storeTT(hash, depth, bestScore, flag, bestMove);

  return bestScore;
}

// ─── Public search API ────────────────────────────────────────────────────────

export interface AIResult {
  move: Move;
  score: number;
  depth: number;
  nodes: number;
}

export async function getBestMove(
  state: GameState,
  timeLimitMs = 1500,
): Promise<AIResult> {
  return new Promise((resolve) => {
    setTimeout(() => {
      const start = Date.now();
      nodesSearched = 0;
      TT.clear();

      const moves = getAllLegalMoves(state);
      if (moves.length === 0) {
        throw new Error('No legal moves');
      }
      if (moves.length === 1) {
        resolve({
          move: moves[0],
          score: 0,
          depth: 0,
          nodes: 1,
        });
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

        const hash = computeZobrist(state);
        const ttMove = getTTMove(hash);
        const ordered = orderMoves(state, moves, ttMove);

        for (const move of ordered) {
          if (Date.now() - start > timeLimitMs) break;

          const next = applyMove(state, move);
          const score = -negamax(next, depth - 1, -beta, -alpha, 1); // ply = 1 at root

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

      resolve({
        move: bestMove,
        score: bestScore,
        depth: completedDepth,
        nodes: nodesSearched,
      });
    }, 10);
  });
}

// ─── NNUE init ────────────────────────────────────────────────────────────────

let initialized = false;

export async function initAI(
  modelUrl: string = '/best_network.nnue',
): Promise<void> {
  if (initialized) return;
  await evaluator.loadWeights(modelUrl);
  initialized = true;
  console.log('AI initialized with trained NNUE model');
}
