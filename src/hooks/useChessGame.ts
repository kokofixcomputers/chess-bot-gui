import { useState, useCallback, useRef } from 'react';
import type { GameState, Move, Piece, Square, Color } from '../engine/types';
import { parseFEN, INITIAL_FEN, getLegalMoves, applyMove, getGameResult, isInCheck, moveToAlg, squareToAlg, pieceColor, pieceType } from '../engine/chess';
import { getBestMove } from '../engine/ai';

export type GamePhase = 'playing' | 'checkmate' | 'stalemate' | 'draw-50' | 'draw-repetition';

export interface HistoryEntry {
  move: Move;
  alg: string;
  fen: string;
  capturedPiece?: Piece;
  aiDepth?: number;
  aiNodes?: number;
}

export interface ChessGameState {
  gameState: GameState;
  selected: Square | null;
  legalMovesForSelected: Move[];
  phase: GamePhase;
  history: HistoryEntry[];
  capturedByWhite: Piece[];
  capturedByBlack: Piece[];
  lastMove: Move | null;
  aiThinking: boolean;
  aiDepth: number;
  aiNodes: number;
  inCheck: boolean;
  promotionPending: { from: Square; to: Square } | null;
  shakeSq: string | null;
}

export interface ChessGameActions {
  selectSquare: (sq: Square) => void;
  completePromotion: (piece: 'Q' | 'R' | 'B' | 'N') => void;
  resetGame: () => void;
  undoMove: () => void;
}

function getPhase(state: GameState): GamePhase {
  const r = getGameResult(state);
  if (r === 'checkmate') return 'checkmate';
  if (r === 'stalemate') return 'stalemate';
  if (r === 'draw-50') return 'draw-50';
  return 'playing';
}

const PLAYER_COLOR: Color = 'w';
const AI_COLOR: Color = 'b';

export function useChessGame(): ChessGameState & ChessGameActions {
  const [gameState, setGameState] = useState<GameState>(() => parseFEN(INITIAL_FEN));
  const [selected, setSelected] = useState<Square | null>(null);
  const [legalMovesForSelected, setLegalMovesForSelected] = useState<Move[]>([]);
  const [phase, setPhase] = useState<GamePhase>('playing');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [capturedByWhite, setCapturedByWhite] = useState<Piece[]>([]);
  const [capturedByBlack, setCapturedByBlack] = useState<Piece[]>([]);
  const [lastMove, setLastMove] = useState<Move | null>(null);
  const [aiThinking, setAiThinking] = useState(false);
  const [aiDepth, setAiDepth] = useState(0);
  const [aiNodes, setAiNodes] = useState(0);
  const [promotionPending, setPromotionPending] = useState<{ from: Square; to: Square } | null>(null);
  const [shakeSq, setShakeSq] = useState<string | null>(null);

  const stateHistoryRef = useRef<GameState[]>([parseFEN(INITIAL_FEN)]);

  const triggerShake = (sq: Square) => {
    const key = `${sq[0]}-${sq[1]}`;
    setShakeSq(key);
    setTimeout(() => setShakeSq(null), 450);
  };

  const triggerAI = useCallback(async (currentState: GameState) => {
    setAiThinking(true);
    try {
      const result = await getBestMove(currentState, 1500);
      const { move } = result;

      const nextState = applyMove(currentState, move);
      const nextPhase = getPhase(nextState);

      stateHistoryRef.current.push(nextState);

      setGameState(nextState);
      setPhase(nextPhase);
      setLastMove(move);
      setAiDepth(result.depth);
      setAiNodes(result.nodes);

      if (move.captured) {
        setCapturedByBlack(prev => [...prev, move.captured!]);
      }

      setHistory(prev => [...prev, {
        move,
        alg: moveToAlg(move),
        fen: '',
        capturedPiece: move.captured,
        aiDepth: result.depth,
        aiNodes: result.nodes,
      }]);
    } catch (err) {
      console.error('AI error:', err);
    } finally {
      setAiThinking(false);
    }
  }, []);

  const executePlayerMove = useCallback((move: Move, currentState: GameState) => {
    const nextState = applyMove(currentState, move);
    const nextPhase = getPhase(nextState);

    stateHistoryRef.current.push(nextState);

    setGameState(nextState);
    setPhase(nextPhase);
    setLastMove(move);
    setSelected(null);
    setLegalMovesForSelected([]);

    if (move.captured) {
      setCapturedByWhite(prev => [...prev, move.captured!]);
    }

    setHistory(prev => [...prev, {
      move,
      alg: moveToAlg(move),
      fen: '',
      capturedPiece: move.captured,
    }]);

    if (nextPhase === 'playing') {
      triggerAI(nextState);
    }
  }, [triggerAI]);

  const selectSquare = useCallback((sq: Square) => {
    if (phase !== 'playing' || aiThinking || promotionPending) return;
    if (gameState.turn !== PLAYER_COLOR) return;

    const [r, c] = sq;
    const piece = gameState.board[r][c];
    const pc = pieceColor(piece);

    // If we have a selection, try to move
    if (selected) {
      const [sr, sc] = selected;

      // Deselect if same square
      if (sr === r && sc === c) {
        setSelected(null);
        setLegalMovesForSelected([]);
        return;
      }

      // Try to move to this square
      const moveCandidates = legalMovesForSelected.filter(
        m => m.to[0] === r && m.to[1] === c
      );

      if (moveCandidates.length > 0) {
        // Check if pawn promotion
        if (moveCandidates.some(m => m.promotion)) {
          setPromotionPending({ from: [sr, sc], to: [r, c] });
          return;
        }
        executePlayerMove(moveCandidates[0], gameState);
        return;
      }

      // Re-select own piece
      if (pc === PLAYER_COLOR) {
        const moves = getLegalMoves(gameState, r, c);
        if (moves.length === 0) {
          triggerShake(sq);
          setSelected(null);
          setLegalMovesForSelected([]);
          return;
        }
        setSelected(sq);
        setLegalMovesForSelected(moves);
        return;
      }

      // Invalid target
      triggerShake(sq);
      setSelected(null);
      setLegalMovesForSelected([]);
      return;
    }

    // Select own piece
    if (pc === PLAYER_COLOR) {
      const moves = getLegalMoves(gameState, r, c);
      if (moves.length === 0) {
        triggerShake(sq);
        return;
      }
      setSelected(sq);
      setLegalMovesForSelected(moves);
    }
  }, [gameState, selected, legalMovesForSelected, phase, aiThinking, promotionPending, executePlayerMove]);

  const completePromotion = useCallback((pt: 'Q' | 'R' | 'B' | 'N') => {
    if (!promotionPending) return;
    const { from, to } = promotionPending;
    const moves = getLegalMoves(gameState, from[0], from[1]);
    const move = moves.find(m => m.to[0] === to[0] && m.to[1] === to[1] && m.promotion === pt);
    if (move) {
      setPromotionPending(null);
      setSelected(null);
      setLegalMovesForSelected([]);
      executePlayerMove(move, gameState);
    }
  }, [promotionPending, gameState, executePlayerMove]);

  const resetGame = useCallback(() => {
    const initial = parseFEN(INITIAL_FEN);
    stateHistoryRef.current = [initial];
    setGameState(initial);
    setSelected(null);
    setLegalMovesForSelected([]);
    setPhase('playing');
    setHistory([]);
    setCapturedByWhite([]);
    setCapturedByBlack([]);
    setLastMove(null);
    setAiThinking(false);
    setAiDepth(0);
    setAiNodes(0);
    setPromotionPending(null);
    setShakeSq(null);
  }, []);

  const undoMove = useCallback(() => {
    // Undo 2 moves (player + AI) to get back to player's turn
    const hist = stateHistoryRef.current;
    if (hist.length < 3) return; // need at least initial + player + ai
    stateHistoryRef.current = hist.slice(0, hist.length - 2);
    const prev = stateHistoryRef.current[stateHistoryRef.current.length - 1];
    setGameState(prev);
    setPhase('playing');
    setSelected(null);
    setLegalMovesForSelected([]);
    setHistory(h => h.slice(0, h.length - 2));
    // Recompute captures from remaining history
    const newHistEntries = history.slice(0, history.length - 2);
    const wb: Piece[] = [], bb: Piece[] = [];
    for (const e of newHistEntries) {
      if (e.capturedPiece) {
        if (e.capturedPiece[0] === 'b') wb.push(e.capturedPiece);
        else bb.push(e.capturedPiece);
      }
    }
    setCapturedByWhite(wb);
    setCapturedByBlack(bb);
    const lastH = newHistEntries[newHistEntries.length - 1];
    setLastMove(lastH?.move ?? null);
  }, [history]);

  const inCheck = isInCheck(gameState, gameState.turn);

  return {
    gameState,
    selected,
    legalMovesForSelected,
    phase,
    history,
    capturedByWhite,
    capturedByBlack,
    lastMove,
    aiThinking,
    aiDepth,
    aiNodes,
    inCheck,
    promotionPending,
    shakeSq,
    selectSquare,
    completePromotion,
    resetGame,
    undoMove,
  };
}
